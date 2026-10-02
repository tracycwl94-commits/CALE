# mmdet/models/roi_heads/iterative_roi_head.py

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import unpack_gt_instances
from .standard_roi_head import StandardRoIHead


class IterMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, use_layer_norm=False):
        super().__init__()

        assert num_layers == 2, "当前checkpoint只支持2层MLP（fc1/fc2）"

        self.layers = nn.ModuleDict({
            "fc1": nn.Linear(input_dim, hidden_dim),
            "fc2": nn.Linear(hidden_dim, output_dim),
        })

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(hidden_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layers["fc1"](x)
        if self.use_layer_norm:
            x = self.norm(x)
        x = self.relu(x)
        x = self.layers["fc2"](x)
        return x


@MODELS.register_module()
class IterativeRoIHead(StandardRoIHead):
    """Iterative RoIHead for MMDetection 3.x.

    The idea is:
    1) Extract RoI features.
    2) Run bbox head once to get cls_score / bbox_pred.
    3) Use cls_score and bbox_pred to generate channel/spatial bias.
    4) Modulate RoI features and iterate.
    """

    def __init__(
        self,
        n_iter: int = 3,
        mlp_layers: int = 2,
        box_reg_iter: int = -1,
        iter_loss_weight: Optional[Sequence[float]] = None,
        reg_iter: int = -1,
        use_layer_norm: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.n_iter = int(n_iter)
        self.mlp_layers = int(mlp_layers)
        self.use_layer_norm = bool(use_layer_norm)

        # Backward-compatible alias
        self.reg_iter = reg_iter if reg_iter != -1 else box_reg_iter
        self.box_reg_iter = box_reg_iter

        if iter_loss_weight is None:
            self.iter_loss_weight = [1.0] * self.n_iter
        else:
            self.iter_loss_weight = list(iter_loss_weight)
            if len(self.iter_loss_weight) != self.n_iter:
                raise ValueError(
                    f'iter_loss_weight length ({len(self.iter_loss_weight)}) '
                    f'must match n_iter ({self.n_iter})'
                )

        # Infer feature dims from the built components.
        # bbox_head.in_channels is the RoI feature channel count in common 3.x heads.
        feat_channels = getattr(self.bbox_head, 'in_channels', 256)
        if isinstance(feat_channels, (list, tuple)):
            feat_channels = feat_channels[0]
        feat_channels = int(feat_channels)

        cls_input_dim = self._infer_cls_input_dim()
        roi_h, roi_w = self._infer_roi_feat_hw()
        loc_output_dim = int(roi_h * roi_w)

        self.cls_sub_module = IterMLP(
            input_dim=cls_input_dim,
            hidden_dim=512,
            output_dim=feat_channels,
            num_layers=self.mlp_layers,
            use_layer_norm=self.use_layer_norm,
        )
        self.loc_sub_module = IterMLP(
            input_dim=4 * getattr(self.bbox_head, 'num_classes', 80),
            hidden_dim=512,
            output_dim=loc_output_dim,
            num_layers=self.mlp_layers,
            use_layer_norm=False,
        )

    def _infer_cls_input_dim(self) -> int:
        """Infer classification branch input dimension."""
        custom_cls_channels = getattr(self.bbox_head, 'custom_cls_channels', False)
        if custom_cls_channels and hasattr(self.bbox_head, 'loss_cls'):
            get_channels = getattr(self.bbox_head.loss_cls, 'get_cls_channels', None)
            if callable(get_channels):
                return int(get_channels(self.bbox_head.num_classes))
        return int(getattr(self.bbox_head, 'num_classes', 80) + 1)

    def _infer_roi_feat_hw(self) -> Tuple[int, int]:
        """Infer RoI feature spatial size, e.g. 7x7."""
        # Most RoI extractors expose roi_layers[0].output_size
        if hasattr(self.bbox_roi_extractor, 'roi_layers') and len(self.bbox_roi_extractor.roi_layers) > 0:
            output_size = getattr(self.bbox_roi_extractor.roi_layers[0], 'output_size', 7)
            if isinstance(output_size, int):
                return output_size, output_size
            if isinstance(output_size, tuple):
                if len(output_size) == 2:
                    return int(output_size[0]), int(output_size[1])
                if len(output_size) == 1:
                    return int(output_size[0]), int(output_size[0])

        # Fallback
        return 7, 7

    def _accumulate_loss_dict(
        self,
        total: dict,
        current: dict,
        weight: float = 1.0,
    ) -> dict:
        """Accumulate tensor-like losses with weight."""
        for k, v in current.items():
            if not torch.is_tensor(v):
                continue
            v = v * weight
            if k in total:
                total[k] = total[k] + v
            else:
                total[k] = v
        return total

    def _bbox_forward(
        self,
        x: Tuple[Tensor],
        rois: Tensor,
        train: bool = False,
    ):
        """Forward bbox branch with iterative feature modulation."""
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        bbox_feats_list = [bbox_feats]
        predictions_list = []

        for _ in range(self.n_iter):
            cls_score, bbox_pred = self.bbox_head(bbox_feats_list[-1])

            predictions_list.append((cls_score, bbox_pred))

            # Generate modulation from current predictions
            # cls_score: [N, Ccls]
            # bbox_pred:  [N, 4 * num_classes] (or similar)
            cur_feats = bbox_feats_list[-1]
            c = cur_feats.size(1)
            h, w = cur_feats.size(-2), cur_feats.size(-1)

            channel_bias = self.cls_sub_module(cls_score).view(-1, c, 1, 1)
            spatial_bias = self.loc_sub_module(bbox_pred).view(-1, 1, h, w)

            next_feats = cur_feats * spatial_bias + channel_bias
            bbox_feats_list.append(next_feats)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            bbox_feats=bbox_feats_list[-1],
        )

        if train:
            return bbox_results, predictions_list
        return bbox_results

    def _bbox_loss_single_legacy(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        iter_id: int,
    ) -> dict:
        """Fallback for bbox heads that still use legacy loss/get_targets APIs."""
        gt_bboxes = [gt_instances.bboxes for gt_instances in batch_gt_instances]
        gt_labels = []
        for gt_instances in batch_gt_instances:
            if hasattr(gt_instances, 'labels'):
                gt_labels.append(gt_instances.labels)
            else:
                gt_labels.append(None)

        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        try:
            loss_dict = self.bbox_head.loss(
                cls_score, bbox_pred, rois, *bbox_targets, iter_id=iter_id)
        except TypeError:
            loss_dict = self.bbox_head.loss(
                cls_score, bbox_pred, rois, *bbox_targets)

        return loss_dict

    def bbox_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
    ) -> dict:
        """BBox loss with iteration-aware aggregation."""
        rois = bbox2roi([res.priors for res in sampling_results])

        bbox_results, predictions_list = self._bbox_forward(x, rois, train=True)

        loss_list = []
        for iter_id, (cls_score, bbox_pred) in enumerate(predictions_list):
            if hasattr(self.bbox_head, 'loss_and_target'):
                # MMDetection 3.x style heads
                loss_and_target = self.bbox_head.loss_and_target(
                    cls_score=cls_score,
                    bbox_pred=bbox_pred,
                    rois=rois,
                    sampling_results=sampling_results,
                    rcnn_train_cfg=self.train_cfg,
                )
                loss_dict = loss_and_target.get('loss_bbox', {})
                if torch.is_tensor(loss_dict):
                    loss_dict = {'loss_bbox': loss_dict}
                elif not isinstance(loss_dict, dict):
                    loss_dict = {'loss_bbox': loss_dict}
            else:
                # Legacy custom head style
                loss_dict = self._bbox_loss_single_legacy(
                    cls_score=cls_score,
                    bbox_pred=bbox_pred,
                    rois=rois,
                    sampling_results=sampling_results,
                    batch_gt_instances=batch_gt_instances,
                    iter_id=iter_id,
                )

            loss_list.append(loss_dict)

        loss_bbox = {}
        for i, loss_dict in enumerate(loss_list):
            weight = float(self.iter_loss_weight[i])
            self._accumulate_loss_dict(loss_bbox, loss_dict, weight)

        # Keep regression loss from a specific iteration if requested.
        if self.reg_iter is not None and self.reg_iter != -1:
            if 0 <= self.reg_iter < len(loss_list) and 'loss_bbox' in loss_list[self.reg_iter]:
                loss_bbox['loss_bbox'] = loss_list[self.reg_iter]['loss_bbox']

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def loss(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: SampleList,
    ) -> dict:
        """MMDetection 3.x training entry."""
        assert len(rpn_results_list) == len(batch_data_samples)

        batch_gt_instances, batch_gt_instances_ignore, _ = unpack_gt_instances(
            batch_data_samples)

        num_imgs = len(batch_data_samples)
        sampling_results = []

        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i], batch_gt_instances_ignore[i])

            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x],
            )
            sampling_results.append(sampling_result)

        losses = dict()

        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results, batch_gt_instances)
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask:
            # Use the inherited MMDet 3.x mask path
            bbox_results = self.bbox_loss(x, sampling_results, batch_gt_instances)
            mask_results = self.mask_loss(
                x,
                sampling_results,
                bbox_results['bbox_feats'],
                batch_gt_instances,
            )
            losses.update(mask_results['loss_mask'])

        return losses
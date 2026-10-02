# Copyright (c) OpenMMLab. All rights reserved.
# CALE score adjustment added to the FRACAL/MMDetection inference path.
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.models.layers import multiclass_nms
from mmdet.models.utils import empty_instances
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from .priors import load_priors


@MODELS.register_module()
class ConvFCCALEBBoxHead(ConvFCBBoxHead):
    """Apply E and L to a C+1-channel ROI classifier before filtering and NMS."""
    def __init__(self, *args, frequency_file, prior_file, eta=0.8, gamma=2., **kwargs):
        super().__init__(*args, **kwargs)
        if self.custom_cls_channels or self.fc_cls.out_features != self.num_classes + 1:
            raise ValueError('This ROI adapter expects an ordinary C+1-channel classifier.')
        self.eta, self.gamma = float(eta), float(gamma)
        bias, prior = load_priors(frequency_file, prior_file, self.num_classes, True)
        self.register_buffer('frequency_bias', bias, persistent=False)
        self.register_buffer('structural_prior', prior, persistent=False)

    def adjust_scores(self, logits):
        p = F.softmax(logits[:, :self.num_classes], dim=-1)
        # x*log(x) has limiting value zero at x=0; normal finite inputs retain
        # the operation order used in the experiments.
        plogp = p * torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))
        omega = -plogp.sum(dim=1) / torch.log(torch.tensor(self.num_classes)) + self.eta
        scores = F.softmax(logits + torch.ger(omega, self.frequency_bias[0]), dim=-1)
        scores = scores / (self.structural_prior ** self.gamma)
        scores /= scores.sum(dim=1, keepdim=True)
        return scores

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]
        scores = self.adjust_scores(cls_score)

        img_shape = img_meta['img_shape']
        num_rois = roi.size(0)
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = roi.repeat_interleave(num_classes, dim=0)
            bbox_pred = bbox_pred.view(-1, self.bbox_coder.encode_size)
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.size(-1) == 4:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        # Get the inside tensor when `bboxes` is a box type
        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.size(-1)
        bboxes = bboxes.view(num_rois, -1)

        if rcnn_test_cfg is None:
            # This means that it is aug test.
            # It needs to return the raw results without nms.
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results

@MODELS.register_module()
class Shared2FCCALEBBoxHead(ConvFCCALEBBoxHead):
    def __init__(self, fc_out_channels=1024, **kwargs):
        super().__init__(num_shared_convs=0, num_shared_fcs=2,
                         num_cls_convs=0, num_cls_fcs=0,
                         num_reg_convs=0, num_reg_fcs=0,
                         fc_out_channels=fc_out_channels, **kwargs)

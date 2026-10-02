# Copyright (c) OpenMMLab. All rights reserved.
# CALE adapter for ATSS. Scores are adjusted before native centerness and NMS.
import copy
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import ATSSHead
from mmdet.registry import MODELS
from .priors import load_priors

def filter_scores_and_topk(scores, score_thr, topk, results=None):
    """Filter class scores while preserving the evaluated top-k and tie ordering."""
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, len(scores))
    if num_topk > 0 and num_topk < len(scores):
        scores, idxs = scores.topk(num_topk)
        valid_idxs = valid_idxs[idxs]
    keep_idxs = valid_idxs[:, 0]
    labels = valid_idxs[:, 1] if valid_idxs.size(1) > 1 else valid_idxs[:, 0] * 0
    if results is not None:
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = value[keep_idxs]
    return scores, labels, keep_idxs, results

def cat_boxes(boxes_list):
    if isinstance(boxes_list[0], torch.Tensor):
        return torch.cat(boxes_list, dim=0)
    from mmdet.structures.bbox import BaseBoxes
    return BaseBoxes.cat(boxes_list)


@MODELS.register_module()
class ATSSCALEHead(ATSSHead):
    def __init__(self, *args, frequency_file, prior_file, eta=0.8, gamma=2., **kwargs):
        super().__init__(*args, **kwargs)
        self.eta, self.gamma = float(eta), float(gamma)
        bias, prior = load_priors(frequency_file, prior_file, self.num_classes, False)
        self.register_buffer('frequency_bias', bias, persistent=False)
        self.register_buffer('structural_prior', prior, persistent=False)

    def adjust_scores(self, logits):
        p = F.softmax(logits, dim=-1) + 1e-9
        entropy = -(p * torch.log(p)).sum(dim=1)
        omega = entropy / torch.log(torch.tensor(self.num_classes, device=logits.device)) + self.eta
        structural = self.structural_prior ** self.gamma
        structural = structural / structural.sum()
        structural_logit = -torch.log10(structural) + torch.log10(
            torch.tensor(1 / self.num_classes, device=logits.device)).item()
        frequency = torch.ger(omega, self.frequency_bias[0])
        return logits.sigmoid() * (logits + frequency + structural_logit).sigmoid()

    def _predict_by_feat_single(self, cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor], img_meta: dict,
                                cfg: ConfigDict, rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        with_score_factors = score_factor_list[0] is not None
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        mlvl_bbox_preds, mlvl_valid_priors, mlvl_scores, mlvl_labels = [], [], [], []
        mlvl_score_factors = [] if with_score_factors else None
        for cls_score, bbox_pred, score_factor, priors in zip(
                cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, self.bbox_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = self.adjust_scores(cls_score)
            scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(
                scores, cfg.get('score_thr', 0), nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            mlvl_bbox_preds.append(filtered_results['bbox_pred'])
            mlvl_valid_priors.append(filtered_results['priors'])
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor[keep_idxs])
        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)
        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)
        # Native postprocess multiplies centerness ONCE after preselection,
        # then uses class-wise NMS and max_per_img. Do not add a sqrt.
        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale,
            with_nms=with_nms, img_meta=img_meta)

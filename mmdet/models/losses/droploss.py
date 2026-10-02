import torch
import torch.nn as nn
import torch.nn.functional as F
from .accuracy import accuracy
from mmdet.registry import MODELS


def get_image_count_frequency(version="v0_5"):
    if version == "v0_5":
        from mmdet.utils.lvis_v0_5_categories import get_image_count_frequency
        return get_image_count_frequency()

    elif version == "v1":
        from mmdet.utils.lvis_v1_0_categories import get_image_count_frequency
        return get_image_count_frequency()

    elif version == "openimage":
        from mmdet.utils.openimage_categories import get_instance_count
        return get_instance_count()

    else:
        raise KeyError(f"{version} not supported")


def logsumexp(x):
    alpha = torch.exp(x)
    return alpha + torch.log1p(-torch.exp(-alpha) + 1e-12)


@MODELS.register_module()
class DropLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_=0.0011,
                 version='v1',
                 use_classif='gumbel',
                 num_classes=1203):

        super().__init__()

        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.lambda_ = lambda_
        self.version = version
        self.use_classif = use_classif
        self.num_classes = num_classes

        self.custom_cls_channels = True
        self.custom_activation = True
        self.custom_accuracy = True

        freq = torch.tensor(
            get_image_count_frequency(version),
            dtype=torch.float32
        )

        # 关键修复
        self.register_buffer("freq_info", freq)

        num_class_included = (self.freq_info < self.lambda_).sum().item()

        print(
            f"DropLoss init: {num_class_included} rare classes included."
        )

    def get_cls_channels(self, num_classes):
        assert num_classes == self.num_classes
        return num_classes

    def get_accuracy(self, cls_score, labels):
        pos_inds = labels < self.num_classes
        acc = accuracy(cls_score[pos_inds], labels[pos_inds])
        return dict(acc_classes=acc)

    def get_activation(self, cls_score):

        if self.use_classif == 'gumbel':
            scores = torch.exp(-torch.exp(-cls_score))
        else:
            scores = torch.sigmoid(cls_score)

        dummy = scores.new_zeros((scores.size(0), 1))
        scores = torch.cat([scores, dummy], dim=1)

        return scores

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        n_i, n_c = cls_score.size()

        self.n_i = n_i
        self.n_c = n_c
        self.gt_classes = label
        self.pred_class_logits = cls_score

        target = cls_score.new_zeros(n_i, n_c + 1)
        inds = torch.arange(n_i, device=cls_score.device)
        target[inds, label] = 1
        target = target[:, :n_c]

        drop_w = 1 - self.threshold_func() * (1 - target)

        if self.use_classif == 'gumbel':

            cls_score = torch.clamp(cls_score, -10, 10)

            cls_loss = (
                torch.exp(-cls_score) * target +
                (target - 1.0) *
                (logsumexp(-cls_score) - torch.exp(-cls_score))
            )

        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_score,
                target,
                reduction='none'
            )

        cls_loss = (cls_loss * drop_w).sum() / n_i

        return self.loss_weight * cls_loss

    def exclude_func_and_ratio(self):

        bg_ind = self.n_c

        fg_mask = (self.gt_classes != bg_ind)

        gt_classes = self.gt_classes[fg_mask]

        if len(gt_classes) == 0:
            ratio = self.freq_info.new_tensor(0.0)
        else:
            ratio = (
                self.freq_info[gt_classes] < self.lambda_
            ).float().mean()

        fg_mask = fg_mask.float().view(self.n_i, 1).expand(self.n_i, self.n_c)

        return fg_mask, ratio

    def threshold_func(self):

        weight = self.pred_class_logits.new_zeros(self.n_c)

        weight[self.freq_info < self.lambda_] = 1

        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)

        fg, ratio = self.exclude_func_and_ratio()

        bg = 1 - fg

        rand = torch.rand_like(bg) * bg

        rand = (rand > ratio).float()

        weight = (rand + fg) * weight

        return weight
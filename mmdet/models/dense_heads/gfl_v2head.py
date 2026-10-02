import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import bbox_overlaps
from mmdet.registry import MODELS
from mmdet.models.dense_heads.gfl_head import GFLHead

@MODELS.register_module()
class GFLV2Head(GFLHead):
    def __init__(self, 
                 num_classes, 
                 in_channels, 
                 reg_max=16, 
                 reg_topk=4, 
                 reg_channels=64, 
                 add_mean=True, 
                 **kwargs):
        # 把 v2 特有的参数存下来
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        
        # 调用父类 (GFL v1) 的初始化
        super(GFLV2Head, self).__init__(num_classes=num_classes, 
                                        in_channels=in_channels, 
                                        reg_max=reg_max, 
                                        **kwargs)

    def _init_layers(self):
        """初始化网络层"""
        # 先让父类把分类和回归的基础卷积层建好
        super(GFLV2Head, self)._init_layers()
        
        # GFLv2 核心：添加 DGQPE (Distribution-Guided Quality Predictor Estimator)
        # 也就是你原来 TF 代码里的 quality_predictor
        
        # 计算输入的维度
        last_dim = self.reg_topk + 1 if self.add_mean else self.reg_topk
        
        # 使用 PyTorch 的 nn.Sequential 重写质量预测器
        self.quality_predictor = nn.Sequential(
            nn.Conv2d(4 * last_dim, self.reg_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.reg_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward_single(self, x, scale):
        """单层特征图的前向传播 (修正版：加入了 scale 参数)"""
        # 1. 经过共享卷积层
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # 2. 基础回归分支预测 (Bounding Box)
        # 注意：这里必须要乘上当前 FPN 层的 scale 缩放系数
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        
        # 3. 基础分类分支预测
        cls_score = self.gfl_cls(cls_feat)

        # --- 以下是 GFLv2 的核心逻辑：利用回归的分布统计特征来指导质量预测 ---
        N, C, H, W = bbox_pred.size()
        # 变形以便提取 topk (C = 4 * (reg_max + 1))
        prob = bbox_pred.view(N, 4, self.reg_max + 1, H, W).softmax(dim=2)
        
        # 提取 top-k 概率
        topk_prob, _ = prob.topk(self.reg_topk, dim=2)
        
        # GFLv2 特性：是否加入均值
        if self.add_mean:
            stat = torch.cat([topk_prob, topk_prob.mean(dim=2, keepdim=True)], dim=2)
        else:
            stat = topk_prob
            
        # 将统计特征输入到质量预测器中
        stat = stat.view(N, -1, H, W)
        quality_score = self.quality_predictor(stat)
        
        # 用预测出的质量分数（IoU 质量）乘以分类分数
        cls_score = cls_score.sigmoid() * quality_score
        
        return cls_score, bbox_pred
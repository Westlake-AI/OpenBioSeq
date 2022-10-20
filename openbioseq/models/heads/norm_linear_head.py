from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cls_head import ClsHead
from ..registry import HEADS


class NormLinear(nn.Linear):

    def __init__(self,
                in_features: int,
                out_features: int,
                bias: bool,
                feature_norm: bool,
                weight_norm: bool):
        super().__init__(in_features, out_features, bias=bias)
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_norm:
            input = F.normalize(input)
        if self.weight_norm:
            weight = F.normalize(self.weight)
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)


@HEADS.register_module()
class NormLinearClsHead(ClsHead):
    """Normalized classifier head.

    Args:
        with_avg_pool (bool): Whether to use GAP before this head.
        loss (dict): Config of classification loss.
        in_channels (int): Number of channels in the input feature map.
        num_classes (int): Number of categories excluding the category.
        feature_norm (bool): Whether to L2-norm the input features.
            Defaults to True.
        weight_norm (bool): Whether to L2-norm the weight in nn.Linear.
            Defaults to False.
        bias (bool): Whether to use bias. Defaults to False.
    """

    def __init__(self,
                 with_avg_pool=False,
                 loss=dict(type='ArcFaceLoss', loss_weight=1.0),
                 num_classes=1000,
                 in_channels=2048,
                 feature_norm=True,
                 weight_norm=False,
                 bias=False,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(NormLinearClsHead, self).__init__(
            with_avg_pool=with_avg_pool, loss=loss, init_cfg=init_cfg, **kwargs)

        assert hasattr('s', self.criterion),\
            'NormLinearClsHead.compute_loss requires `s` like ArcFaceLoss.'

        self.num_classes = num_classes
        self.in_channels = in_channels

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = NormLinear(self.in_channels, self.num_classes, bias,
                             feature_norm, weight_norm)

    def forward(self, x):
        assert isinstance(x, (tuple, list)) and len(x) == 1
        if self.fc is None:
            return x
        x = x[0]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        return [self.fc(x)]

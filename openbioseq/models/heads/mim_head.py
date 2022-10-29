import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule
from mmcv.cnn.utils.weight_init import trunc_normal_init

from ..builder import build_loss
from ..registry import HEADS
from .cls_head import ClsHead


@HEADS.register_module
class MAEPretrainHead(BaseModule):
    """Pre-training head for MAE.

    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
        feature_Nd (str): Build Nd feature in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self,
                 norm_pix=False,
                 patch_size=16,
                 feature_Nd="2d",
                 init_cfg=None):
        super(MAEPretrainHead, self).__init__(init_cfg)
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.feature_Nd = feature_Nd
        assert feature_Nd in ["2d", "1d",]

    def patchify(self, data):
        p = self.patch_size
        if self.feature_Nd == "2d":  # (B, C, H, W)
            assert data.shape[2] == data.shape[3] and data.shape[2] % p == 0
            h = w = data.shape[2] // p
            x = data.reshape(shape=(data.shape[0], 3, h, p, w, p))
            x = torch.einsum('nchpwq->nhwpqc', x)
            x = x.reshape(shape=(data.shape[0], h * w, p**2 * 3))
        else:  # (B, L, C)
            assert data.shape[1] % p == 0

        return x

    def forward(self, x, pred, mask):
        losses = dict()
        target = self.patchify(x)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        losses['loss'] = loss
        return losses


@HEADS.register_module()
class MAEFinetuneHead(ClsHead):
    """Fine-tuning head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, **kwargs):
        super(MAEFinetuneHead, self).__init__(**kwargs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=2e-5, bias=0)

    def forward(self, x):
        """"Get the logits."""
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        return [self.fc(x)]


@HEADS.register_module()
class MAELinprobeHead(ClsHead):
    """Linear probing head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, in_channels=786, **kwargs):
        super(MAELinprobeHead, self).__init__(in_channels=in_channels, **kwargs)
        self.bn = nn.BatchNorm1d(in_channels, affine=False, eps=1e-6)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.01, bias=0)

    def forward(self, x):
        """"Get the logits."""
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = self.bn(x[0])
        return [self.fc(x)]


@HEADS.register_module
class SimMIMHead(BaseModule):
    """Pretrain Head for SimMIM.

    Args:
        encoder_in_channels (int): Number of input channels for encoder.
        feature_Nd (str): Build Nd feature in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self, encoder_in_channels=3, feature_Nd="2d", init_cfg=None):
        super(SimMIMHead, self).__init__(init_cfg)
        self.encoder_in_channels = encoder_in_channels
        self.feature_Nd = feature_Nd
        assert feature_Nd in ["2d", "1d",]

    def forward(self, x, x_rec, mask):
        if self.feature_Nd == "2d":  # (B, C, H, W)
            scale_h, scale_w = x.size(2) / mask.size(1), x.size(3) / mask.size(2)
            if scale_h > 1:
                mask = mask.repeat_interleave(int(scale_h), 1).repeat_interleave(
                    int(scale_w), 2).unsqueeze(1).contiguous()
            else:
                mask = F.interpolate(mask.type_as(x).unsqueeze(1),
                                    scale_factor=(scale_h, scale_w), mode="nearest")
        else:  # (B, L, C) -> (B x L, C)
            assert x.size(1) == mask.size(1) or mask.size(1) == 1

        loss_rec = F.l1_loss(x_rec, x, reduction='none')
        loss = (loss_rec * mask).sum() / (mask.sum() +
                                          1e-5) / self.encoder_in_channels
        losses = dict(loss=loss)

        return losses


@HEADS.register_module
class MIMHead(BaseModule):
    """Head for Masked Image or Language Modeling training.

    Args:
        loss (dict): Config of regression loss.
        encoder_in_channels (int): Number of input channels for encoder.
        unmask_weight (float): Loss weight for unmasked patches.
        feature_Nd (str): Build Nd feature in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self,
                 loss=dict(
                    type='RegressionLoss', loss_weight=1.0, mode="l1_loss"),
                 encoder_in_channels=3,
                 unmask_weight=0,
                 feature_Nd="2d",
                 init_cfg=None,
                 **kwargs):
        super(MIMHead, self).__init__(init_cfg)
        self.encoder_in_channels = encoder_in_channels
        self.unmask_weight = unmask_weight
        self.feature_Nd = feature_Nd
        assert feature_Nd in ["2d", "1d",]
        assert loss is None or isinstance(loss, dict)

        if loss is None:
            loss = dict(
                type='RegressionLoss', loss_weight=1.0, mode="l1_loss")
        self.criterion = build_loss(loss)

    def forward(self, x, x_rec, mask):
        if self.feature_Nd == "2d":  # (B, C, H, W)
            scale_h, scale_w = x.size(2) / mask.size(1), x.size(3) / mask.size(2)
            if scale_h > 1:
                mask = mask.repeat_interleave(int(scale_h), 1).repeat_interleave(
                    int(scale_w), 2).unsqueeze(1).contiguous()
            else:
                mask = F.interpolate(mask.type_as(x).unsqueeze(1),
                                    scale_factor=(scale_h, scale_w), mode="nearest")
        else:  # (B, L, C) -> (B x L, C)
            assert x.size(1) == mask.size(1) or mask.size(1) == 1

        # loss
        if self.unmask_weight > 0.:
            # reweight unmasked patches
            mask_s = mask.clone()
            mask_s = mask_s + (1. - mask_s) * self.unmask_weight
        else:
            mask_s = mask
        loss_rec = self.criterion(x_rec, x, reduction_override='none')
        loss_rec = (loss_rec * mask_s).sum() / (
            (mask_s.sum() + 1e-5) * self.encoder_in_channels)
        losses = dict(loss=loss_rec)

        return losses

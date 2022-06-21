import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def make_plain_layer(in_channels,
                     out_channels,
                     kernel_size,
                     num_blocks,
                     conv_cfg=None,
                     norm_cfg=None,
                     act_cfg=dict(type='ReLU'),
                     dilation=1,
                     dropout=0.,
                    ):
    layers = []
    for _ in range(num_blocks):
        layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=kernel_size // 2,
            bias='auto',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        layers.append(layer)
        in_channels = out_channels
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
    if dropout > 0.:
        layers.append(nn.Dropout(dropout))

    return layers


@BACKBONES.register_module()
class PlainCNN(BaseBackbone):
    """`AlexNet <https://en.wikipedia.org/wiki/AlexNet>`_ backbone.

    The input for AlexNet is a 224x224 RGB image.

    Args:
        stage
        mlp_neck (dict): additional MLP neck in SSL. Default is None.
    """
    arch_settings = {
        5:  (1, 1, 1, 1),
        6:  (1, 1, 1, 2),
        7:  (1, 1, 2, 2),
        8:  (2, 2, 2, 2),
        10: (2, 2, 3, 3),
        12: (2, 2, 4, 4),
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 patch_size=7,
                 kernel_size=7,
                 patchfied=False,
                 base_channels=64,
                 out_channels=512,
                 out_indices=(3,),
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 drop_rate=0.0,
                 norm_eval=False,
                 pretrained=None):
        super(PlainCNN, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # build stem
        self.stem = ConvModule(
            in_channels=in_channels,
            out_channels=base_channels // 2,
            kernel_size=patch_size,
            stride=patch_size if patchfied else min(4, patch_size // 2),
            padding=patch_size // 2,
            bias='auto',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # build plain cnn
        self.stage_blocks = self.arch_settings[depth]
        self.range_sub_modules = []
        self.plain_layers = []
        start_idx = 0
        in_channels = base_channels // 2
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks + 1
            end_idx = start_idx + num_modules
            out_channels = self.base_channels * 2**i if i < 3 else self.out_channels
            plain_layer = make_plain_layer(
                in_channels,
                out_channels,
                self.kernel_size,
                num_blocks,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dilation=1,
                dropout=drop_rate)
            in_channels = out_channels
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, nn.Sequential(*plain_layer))
            self.plain_layers.append(layer_name)

        self.init_weights(pretrained=pretrained)
        self._freeze_stages()
    
    def init_weights(self, pretrained=None):
        super(PlainCNN, self).init_weights(pretrained)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                normal_init(m, std=0.01, bias=0)
            elif isinstance(m, (_BatchNorm, nn.LayerNorm)):
                constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in self.stem.modules():
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, layer_name in enumerate(self.plain_layers):
            plain_layer = getattr(self, layer_name)
            x = plain_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def train(self, mode=True):
        super(PlainCNN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

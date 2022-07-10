from typing import Sequence

import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init
from mmcv.runner.base_module import ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..utils import PatchEmbed1d
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@BACKBONES.register_module()
class SequenceLSTM(BaseBackbone):
    """General Sequence LSTM.

    Args:
        seq_len (int): The expected input sequence length. Because we support
            dynamic input length, just set the argument to the most common
            input shape. Defaults to None.
        patch_size (int): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        embed_dims (int): Override of embed_dim in LSTM. Defaults to 64.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        stop_grad_conv1 (bool): Whether to stop grad of conv1 in PatchEmbed (for
            MoCo.V3 design). Defaults to False.
    """

    def __init__(self,
                 seq_len=128,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=64,
                 num_layers=2,
                 out_indices=-1,
                 drop_rate=0.,
                 bidirectional=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=False,
                 patch_cfg=dict(),
                 stop_grad_conv1=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 **kwargs):
        super(SequenceLSTM, self).__init__(init_cfg)

        self.embed_dims = max(1, embed_dims // 2)
        self.num_layers = num_layers
        self.seq_len = seq_len if seq_len is not None else 64
        self.patch_size = patch_size
        self.frozen_stages = frozen_stages
        self.init_cfg = init_cfg

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=seq_len,
            embed_dims=self.embed_dims,
            conv_type='Conv1d',
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="GELU"),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed1d(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        self.num_patches = self.patch_embed.init_out_size
        self.drop_after_embed = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # build layers
        self.layers = ModuleList()
        for i in range(self.num_layers):
            _layer = nn.LSTM(
                input_size=self.embed_dims,
                num_layers=1,
                hidden_size=self.embed_dims * 2,
                batch_first=True,
                dropout=drop_rate,
                bidirectional=bidirectional)
            self.layers.append(_layer)
            self.embed_dims = int(2 * self.embed_dims * (1 + bidirectional))
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)
        
        # freeze stages
        self.stop_grad_conv1 = stop_grad_conv1
        if isinstance(self.patch_embed, PatchEmbed1d):
            if stop_grad_conv1:
                self.patch_embed.projection.weight.requires_grad = False
                self.patch_embed.projection.bias.requires_grad = False
        self._freeze_stages()
    
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    def init_weights(self, pretrained=None):
        super(SequenceLSTM, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    trunc_normal_init(m, std=0.02, bias=0)
                elif isinstance(m, (
                    nn.LayerNorm, _BatchNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def forward(self, x):
        x, seq_len = self.patch_embed(x)
        x = self.drop_after_embed(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                x = x.reshape(B, seq_len, C)
                x = x.permute(0, 2, 1)  # (B, C, L)
                outs.append(x)

        return outs

    def _freeze_stages(self):
        """Freeze patch_embed layer, some parameters and stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i == (self.num_layers) and self.final_norm:
                for param in getattr(self, 'norm1').parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SequenceLSTM, self).train(mode)
        self._freeze_stages()

import math
from typing import Sequence
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init, \
                                       uniform_init, xavier_init
from mmcv.runner.base_module import ModuleList
from mmcv.utils.parrots_wrapper import _BatchNorm

from openbioseq.utils import get_root_logger, print_log
from ..utils import resize_pos_embed, PatchEmbed1d, ConvPatchEmbed1d, \
                    build_1d_sincos_position_embedding
from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .vision_transformer import TransformerEncoderLayer


@BACKBONES.register_module()
class SequenceTransformer(BaseBackbone):
    """General Sequence Transformer.

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.
            Default: 'base'
        seq_len (int): The expected input sequence length. Because we support
            dynamic input length, just set the argument to the most common
            input shape. Defaults to None.
        patch_size (int): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        embed_dims (int): Override of embed_dim. Defaults to None.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        feat_scale (bool): If True, use FeatScale (anti-oversmoothing).
            FeatScale re-weights feature maps on separate frequency bands
            to amplify the high-frequency signals.
            Defaults to False.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            ``with_cls_token`` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "linear".
        init_values (float, optional): The init value of gamma in
            TransformerEncoderLayer. Defaults to 0.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        stop_grad_conv1 (bool): Whether to stop grad of conv1 in PatchEmbed (for
            MoCo.V3 design). Defaults to False.
        fix_pos_embed (bool): Whether to use learnable pos_embed or fixed sin-cos
            pos_embed. Defaults to True.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['t', 'tiny',], {
                'embed_dims': 384,
                'num_layers': 6,
                'num_heads': 6,
                'feedforward_channels': 384 * 4,
            }),
        **dict.fromkeys(  # ViT-like
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(  # ViT-like
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(  # ViT-like
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(  # DeiT-like
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(  # DeiT-like
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(  # DeiT-like
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
        **dict.fromkeys(  # MoCo.V3 ViT settings
            ['mocov3-s', 'mocov3-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 1536,
            }),
    }

    def __init__(self,
                 arch='base',
                 seq_len=128,
                 patch_size=16,
                 patchfied=True,
                 in_channels=3,
                 embed_dims=None,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 feat_scale=False,
                 attn_scale=False,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 stem_layer=1,
                 final_norm=True,
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='linear',
                 init_values=0.0,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 stop_grad_conv1=False,
                 fix_pos_embed=False,
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None,
                 **kwargs):
        super(SequenceTransformer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
            self.arch = 'deit'

        if embed_dims is not None:
            ffn_dim = embed_dims * self.arch_settings['feedforward_channels'] / \
                self.arch_settings['embed_dims']
            self.arch_settings['embed_dims'] = embed_dims
            self.arch_settings['feedforward_channels'] = ffn_dim
        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.seq_len = seq_len if seq_len is not None else 64
        self.patch_size = patch_size
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.init_cfg = init_cfg

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=seq_len,
            embed_dims=self.embed_dims,
            conv_type='Conv1d',
            kernel_size=patch_size,
            stride=patch_size if patchfied else patch_size // 2,
        )
        if stem_layer <= 1:
            _patch_cfg.update(patch_cfg)
            self.patch_embed = PatchEmbed1d(**_patch_cfg)
        else:
            _patch_cfg.update(dict(
                num_layers=stem_layer,
                act_cfg=act_cfg,
            ))
            _patch_cfg.update(patch_cfg)
            self.patch_embed = ConvPatchEmbed1d(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        self.num_patches = self.patch_embed.init_out_size

        # Set cls token
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        if with_cls_token and output_cls_token:
            self.num_extra_tokens = 1  # cls_token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        else:
            self.cls_token = None
            self.num_extra_tokens = 0

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.num_extra_tokens, self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

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

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                init_values=init_values,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                feat_scale=feat_scale,
                attn_scale=attn_scale,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)
        
        # freeze stages
        self.stop_grad_conv1 = stop_grad_conv1
        self.fix_pos_embed = fix_pos_embed
        if isinstance(self.patch_embed, PatchEmbed1d):
            if stop_grad_conv1:
                self.patch_embed.projection.weight.requires_grad = False
                self.patch_embed.projection.bias.requires_grad = False
        if fix_pos_embed:
            pos_emb = build_1d_sincos_position_embedding(
                patches_size=self.patch_resolution, embed_dims=self.embed_dims,
                cls_token=True)
            self.pos_embed.data.copy_(pos_emb)
            self.pos_embed.requires_grad = False
        self._freeze_stages()
    
    @property
    def norm1(self):
        return getattr(self, self.norm1_name)
    
    def init_weights(self, pretrained=None):
        super(SequenceTransformer, self).init_weights(pretrained)

        if pretrained is None:
            if "mocov3" not in self.arch:  # normal ViT
                if self.init_cfg is None:
                    for m in self.modules():
                        if isinstance(m, (nn.Conv1d, nn.Linear)):
                            trunc_normal_init(m, std=0.02, bias=0)
                        elif isinstance(m, (
                            nn.LayerNorm, _BatchNorm, nn.GroupNorm, nn.SyncBatchNorm)):
                            constant_init(m, val=1, bias=0)
                # pos_embed & cls_token
                if not self.fix_pos_embed:
                    nn.init.trunc_normal_(self.pos_embed, mean=0, std=.02)
                if self.cls_token is not None:
                    nn.init.trunc_normal_(self.cls_token, mean=0, std=.02)
            else:  # MoCo.V3 pre-training
                # Use fixed 1D sin-cos position embedding
                pos_emb = build_1d_sincos_position_embedding(
                    patches_size=self.patch_resolution,
                    embed_dims=self.embed_dims,
                    cls_token=True)
                self.pos_embed.data.copy_(pos_emb)
                self.pos_embed.requires_grad = False
                # xavier_uniform initialization for PatchEmbed1d
                if isinstance(self.patch_embed, PatchEmbed1d):
                    val = math.sqrt(
                        6. / float(3 * reduce(mul, self.patch_size, 1) + self.embed_dims))
                    uniform_init(self.patch_embed.projection, -val, val, bias=0)
                # initialization for linear layers
                for name, m in self.named_modules():
                    if isinstance(m, nn.Linear):
                        if 'qkv' in name:  # treat the weights of Q, K, V separately
                            val = math.sqrt(
                                6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                            uniform_init(m, -val, val, bias=0)
                        else:
                            xavier_init(m, distribution='uniform')
                if self.cls_token is not None:
                    nn.init.normal_(self.cls_token, std=1e-6)
    
    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return
        
        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            logger = get_root_logger()
            print_log(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.',
                logger=logger)
            ckpt_pos_embed_shape = int(ckpt_pos_embed_shape[1] - self.num_extra_tokens)
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def forward(self, x):
        B = x.shape[0]
        x, seq_len = self.patch_embed(x)
        
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + resize_pos_embed(
            self.pos_embed,
            src_shape=self.patch_resolution,
            dst_shape=seq_len,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                if self.with_cls_token:
                    patch_token = x[:, 1:].reshape(B, seq_len, C)
                    patch_token = patch_token.permute(0, 2, 1)
                    cls_token = x[:, 0]
                else:
                    patch_token = x.reshape(B, seq_len, C)
                    patch_token = patch_token.permute(0, 2, 1)  # (B, C, N)
                    cls_token = None
                if self.output_cls_token:
                    out = [patch_token, cls_token]
                else:
                    out = patch_token
                outs.append(out)

        return outs

    def _freeze_stages(self):
        """Freeze patch_embed layer, some parameters and stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            
            if self.cls_token is not None:
                self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i == (self.num_layers) and self.final_norm:
                for param in getattr(self, 'norm1').parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(SequenceTransformer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (_BatchNorm, nn.SyncBatchNorm)):
                    m.eval()

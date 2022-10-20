# Reference: https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/attention.py
from typing import Sequence
import warnings

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, ConvModule)
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule

from .helpers import to_2tuple


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        super(WindowMSA, self).init_weights()

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shift Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults to dict(type='DropPath', drop_prob=0.).
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 pad_small_map=False,
                 input_resolution=None,
                 auto_pad=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if input_resolution is not None or auto_pad is not None:
            warnings.warn(
                'The ShiftWindowMSA in new version has supported auto padding '
                'and dynamic input shape in all condition. And the argument '
                '`auto_pad` and `input_resolution` have been deprecated.',
                DeprecationWarning)

        self.shift_size = shift_size
        self.window_size = window_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.drop = build_dropout(dropout_layer)
        self.pad_small_map = pad_small_map

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, f"The query length {L} doesn't match the input "\
            f'shape ({H}, {W}).'
        query = query.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size

        if min(H, W) == window_size:
            # If not pad small feature map, avoid shifting when the window size
            # is equal to the size of feature map. It's to align with the
            # behavior of the original implementation.
            shift_size = shift_size if self.pad_small_map else 0
        elif min(H, W) < window_size:
            # In the original implementation, the window size will be shrunk
            # to the size of feature map. The behavior is different with
            # swin-transformer for downstream tasks. To support dynamic input
            # shape, we don't allow this feature.
            assert self.pad_small_map, \
                f'The input shape ({H}, {W}) is smaller than the window ' \
                f'size ({window_size}). Please set `pad_small_map=True`, or ' \
                'decrease the `window_size`.'

        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))

        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if shift_size > 0:
            query = torch.roll(
                query, shifts=(-shift_size, -shift_size), dims=(1, 2))

        attn_mask = self.get_attn_mask((H_pad, W_pad),
                                       window_size=window_size,
                                       shift_size=shift_size,
                                       device=query.device)

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(query, window_size)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad,
                                        window_size)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if H != H_pad or W != W_pad:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)

        return x

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    @staticmethod
    def get_attn_mask(hw_shape, window_size, shift_size, device=None):
        if shift_size > 0:
            img_mask = torch.zeros(1, *hw_shape, 1, device=device)
            h_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = ShiftWindowMSA.window_partition(
                img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    `attn_scale` is modified from : `Anti-Oversmoothing in Deep Vision
    Transformers via the Fourier Domain Analysis: From Theory to Practice
    <https://arxiv.org/abs/2203.05962>`_

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 attn_scale=False,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_scale = attn_scale
        if self.attn_scale:
            self.lamb = nn.Parameter(
                torch.zeros(num_heads), requires_grad=True)

        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self.attn_scale:
            attn_d = torch.ones(
                attn.shape[-2:], device=attn.device) / N  # [l, l]
            attn_d = attn_d[None, None, ...]  # [B, N, l, l]
            attn_h = attn - attn_d  # [B, N, l, l]
            attn_h = attn_h * (1. + self.lamb[None, :, None, None]
                               )  # [B, N, l, l]
            attn = attn_d + attn_h  # [B, N, l, l]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class MultiheadAttentionWithRPE(MultiheadAttention):
    """Multi-head Attention Module with relative position.

    This module rewrite the MultiheadAttention in MMSelfSup by adding the
    relative position bias.

    `attn_scale` is modified from : `Anti-Oversmoothing in Deep Vision
    Transformers via the Fourier Domain Analysis: From Theory to Practice
    <https://arxiv.org/abs/2203.05962>`_

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        window_size (int): The window size of the relative position bias.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        attn_scale (bool): If True, use AttnScale (anti-oversmoothing).
            AttnScale decomposes a self-attention block into low-pass and
            high-pass components, then rescales and combines these two filters
            to produce an all-pass self-attention matrix.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 window_size: int,
                 input_dims: int = None,
                 attn_drop: float = 0,
                 proj_drop: float = 0,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 proj_bias: bool = True,
                 attn_scale: bool = False,
                 init_cfg: dict = None) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            input_dims=input_dims,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_bias=proj_bias,
            attn_scale=attn_scale,
            init_cfg=init_cfg)

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        assert isinstance(window_size, Sequence)
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        # relative_position_bias_table shape is (2*Wh-1 * 2*Ww-1 + 3, nH)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for
        # each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # coords shape is (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)

        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index',
                             relative_position_index)

        self.attn_scale = attn_scale
        if self.attn_scale:
            self.lamb = nn.Parameter(
                torch.zeros(num_heads), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        B, N, _ = x.shape
        qkv = F.linear(
            x, weight=self.qkv.weight,
            bias=qkv_bias).reshape(B, N, 3, self.num_heads,
                                   self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[
                    self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)

        if self.attn_scale:
            attn_d = torch.ones(
                attn.shape[-2:], device=attn.device) / N  # [l, l]
            attn_d = attn_d[None, None, ...]  # [B, N, l, l]
            attn_h = attn - attn_d  # [B, N, l, l]
            attn_h = attn_h * (1. + self.lamb[None, :, None, None]
                               )  # [B, N, l, l]
            attn = attn_d + attn_h  # [B, N, l, l]

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ConditionalPositionEncoding(BaseModule):
    """The Conditional Position Encoding (CPE) module.

    The CPE is the implementation of 'Conditional Positional Encodings
    for Vision Transformers <https://arxiv.org/abs/2102.10882>'_.

    Args:
       in_channels (int): Number of input channels.
       embed_dims (int): The feature dimension. Default: 768.
       stride (int): Stride of conv layer. Default: 1.
    """

    def __init__(self, in_channels, embed_dims=768, stride=1, init_cfg=None):
        super(ConditionalPositionEncoding, self).__init__(init_cfg=init_cfg)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dims,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
            groups=embed_dims)
        self.stride = stride

    def forward(self, x, hw_shape=None):
        B, N, C = x.shape
        feat_token = x
        if hw_shape is not None:
            H, W = hw_shape
            # convert (B, N, C) to (B, C, H, W)
            feat_token = feat_token.transpose(1, 2).view(B, C, H, W)
        else:
            # convert (B, N, C) to (B, C, N)
            feat_token = feat_token.transpose(1, 2)
        if self.stride == 1:
            x = self.proj(feat_token) + feat_token
        else:
            x = self.proj(feat_token)
        x = x.flatten(2).transpose(1, 2)
        return x


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W) or (N,).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W) or (N,).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    if not (isinstance(src_shape, int) and isinstance(dst_shape, int)):
        if len(src_shape) == 2 and len(dst_shape) == 2:
            # 2D img of (H, W)
            if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
                return pos_embed
            _, L, C = pos_embed.shape
            src_h, src_w = src_shape
            assert L == src_h * src_w + num_extra_tokens, \
                f"The length of `pos_embed` ({L}) doesn't match the expected " \
                f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
                '`img_size` argument.'
            extra_tokens = pos_embed[:, :num_extra_tokens]
            src_weight = pos_embed[:, num_extra_tokens:]
            src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Requre (H, W) in {src_shape} and {dst_shape}.")
    else:
        # 1D seq of (L,)
        mode = 'linear'
        if src_shape == dst_shape:
            return pos_embed
        _, L, C = pos_embed.shape
        assert L == src_shape + num_extra_tokens, \
            f"The length of `pos_embed` ({L}) doesn't match the expected " \
            f"shape ({src_shape}+{num_extra_tokens})."
        extra_tokens = pos_embed[:, :num_extra_tokens]
        src_weight = pos_embed[:, num_extra_tokens:]
        src_weight = src_weight.reshape(1, src_shape, C).permute(0, 2, 1)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


def resize_relative_position_bias_table(src_shape, dst_shape, table, num_head):
    """Resize relative position bias table.

    Args:
        src_shape (int): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (int): The resolution of downsampled new training
            image, in format (H, W).
        table (tensor): The relative position bias of the pretrained model.
        num_head (int): Number of attention heads.

    Returns:
        torch.Tensor: The resized relative position bias table.
    """
    from scipy import interpolate

    def geometric_progression(a, r, n):
        return a * (1.0 - r**n) / (1.0 - r)

    left, right = 1.01, 1.5
    while right - left > 1e-6:
        q = (left + right) / 2.0
        gp = geometric_progression(1, q, src_shape // 2)
        if gp > dst_shape // 2:
            right = q
        else:
            left = q

    dis = []
    cur = 1
    for i in range(src_shape // 2):
        dis.append(cur)
        cur += q**(i + 1)

    r_ids = [-_ for _ in reversed(dis)]

    x = r_ids + [0] + dis
    y = r_ids + [0] + dis

    t = dst_shape // 2.0
    dx = np.arange(-t, t + 0.1, 1.0)
    dy = np.arange(-t, t + 0.1, 1.0)

    all_rel_pos_bias = []

    for i in range(num_head):
        z = table[:, i].view(src_shape, src_shape).float().numpy()
        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
        all_rel_pos_bias.append(
            torch.Tensor(f_cubic(dx,
                                 dy)).contiguous().view(-1,
                                                        1).to(table.device))
    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
    return new_rel_pos_bias


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        img_size (int | tuple): The size of input image. Default: 224
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None
    """

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 embed_dims=768,
                 norm_cfg=None,
                 conv_cfg=None,
                 init_cfg=None):
        super(PatchEmbed, self).__init__(init_cfg)
        warnings.warn('The `PatchEmbed` in mmcls will be deprecated. '
                      'Please use `mmcv.cnn.bricks.transformer.PatchEmbed`. '
                      "It's more general and supports dynamic input shape")

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.embed_dims = embed_dims

        # Use conv layer to embed
        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=16, stride=16, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, in_channels, embed_dims)

        # Calculate how many patches a input image is splited to.
        h_out, w_out = [(self.img_size[i] + 2 * self.projection.padding[i] -
                         self.projection.dilation[i] *
                         (self.projection.kernel_size[i] - 1) - 1) //
                        self.projection.stride[i] + 1 for i in range(2)]

        self.patches_resolution = (h_out, w_out)
        self.num_patches = h_out * w_out

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        # The output size is (B, N, D), where N=H*W/P/P, D is embid_dim
        x = self.projection(x).flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class AdaptivePadding1d(nn.Module):
    """Applies padding adaptively to the input sequence.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int ): Size of the kernel. Default: 1.
        stride (int ): Stride of the filter. Default: 1.
        dilation (int ): Spacing between kernel elements. Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super(AdaptivePadding1d, self).__init__()
        assert padding in ('same', 'corner')
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, seq_len):
        """Calculate the padding size of input.

        Args:
            input_shape (`torch.Size` or int): arrange as N or (N,).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        if not isinstance(seq_len, int):
            seq_len = seq_len[0]
        output = math.ceil(seq_len / self.stride)
        pad = max((output - 1) * self.stride +
                  (self.kernel_size - 1) * self.dilation + 1 - seq_len, 0)
        return pad

    def forward(self, x):
        """ Add adaptive padding to `x` in (B, C, N) """
        pad = self.get_pad_shape(x.size(2))
        if pad > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad])
            elif self.padding == 'same':
                x = F.pad(x, [pad // 2, pad - pad // 2])
        return x


class PatchEmbed1d(BaseModule):
    """Data to Patch Embedding for 1D Sequence.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of embedding
            conv. When it is a string, it means the mode of full padding,
            support "same" and "corner" now. Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer.
            Default: None.
        input_size (int | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv1d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 act_cfg=None,
                 input_size=None,
                 init_cfg=None,
                 **kwargs):
        super(PatchEmbed1d, self).__init__(init_cfg)
        
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding1d(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        
        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1] \
            if norm_cfg is not None else None
        self.activate = build_activation_layer(act_cfg) \
            if act_cfg is not None else None

        if input_size:
            assert isinstance(input_size, int), f"1D Sequence length {input_size}."
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad = self.adaptive_padding.get_pad_shape(input_size)
                input_size += + pad
            
            self.init_out_size = (input_size + 2 * padding - dilation *
                                 (kernel_size - 1) - 1) // stride + 1
        else:
            self.init_input_size = None
            self.init_out_size = None
        self.num_patches = self.init_out_size

    def forward(self, x):
        if self.adaptive_padding:
            x = self.adaptive_padding(x)
        
        x = self.projection(x)
        out_len = x.shape[2]
        x = x.transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x, out_len


class ConvPatchEmbed1d(BaseModule):
    """Data to Patch Embedding for 1D Sequence.

    Args:
        in_channels (int): The num of input channels. Default: 3.
        embed_dims (int): The dimensions of embedding. Default: 768.
        num_layers (int): The number of convolution layers. Default: 2.
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of embedding
            conv. When it is a string, it means the mode of full padding,
            support "same" and "corner" now. Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer.
            Default: None.
        input_size (int | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=2,
                 conv_type='Conv1d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 act_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super(ConvPatchEmbed1d, self).__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        if stride is None:
            stride = kernel_size
        
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding1d(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        
        layers = [
            ConvModule(
                in_channels=in_channels if i == 0 else \
                    embed_dims // 2 ** (-i + self.num_layers),
                out_channels=embed_dims // 2 ** (-i + self.num_layers-1),
                kernel_size=kernel_size,
                stride=stride if i == 0 else 1,
                padding=padding if i == 0 else kernel_size // 2,
                bias=bias,
                conv_cfg=dict(type=conv_type),
                norm_cfg=norm_cfg,
                act_cfg=act_cfg if i != self.num_layers - 1 else None,
            ) for i in range(self.num_layers)
        ]
        self.projection = nn.Sequential(*layers)

        if input_size:
            assert isinstance(input_size, int), f"1D Sequence length {input_size}."
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad = self.adaptive_padding.get_pad_shape(input_size)
                input_size += + pad
            
            self.init_out_size = (input_size + 2 * padding - dilation *
                                 (kernel_size - 1) - 1) // stride + 1
        else:
            self.init_input_size = None
            self.init_out_size = None
        self.num_patches = self.init_out_size

    def forward(self, x):
        if self.adaptive_padding:
            x = self.adaptive_padding(x)
        
        x = self.projection(x)
        out_len = x.shape[2]
        x = x.transpose(1, 2)
        return x, out_len


# Modified from pytorch-image-models
class HybridEmbed(BaseModule):
    """CNN Feature Map Embedding.

    Extract feature map from CNN, flatten,
    project to embedding dim.

    Args:
        backbone (nn.Module): CNN backbone
        img_size (int | tuple): The size of input image. Default: 224
        feature_size (int | tuple, optional): Size of feature map extracted by
            CNN backbone. Default: None
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_cfg (dict, optional): The config dict for conv layers.
            Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_channels=3,
                 embed_dims=768,
                 conv_cfg=None,
                 init_cfg=None):
        super(HybridEmbed, self).__init__(init_cfg)
        assert isinstance(backbone, nn.Module)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of
                #  determining the exact dim of the output feature
                #  map for all networks, the feature metadata has
                #  reliable channel and stride info, but using
                #  stride to calc feature dim requires info about padding of
                #  each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_channels, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]

        # Use conv layer to embed
        conv_cfg = conv_cfg or dict()
        _conv_cfg = dict(
            type='Conv2d', kernel_size=1, stride=1, padding=0, dilation=1)
        _conv_cfg.update(conv_cfg)
        self.projection = build_conv_layer(_conv_cfg, feature_dim, embed_dims)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            # last feature if backbone outputs list/tuple of features
            x = x[-1]
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer use nn.Unfold to group feature map by kernel_size, and use norm
    and linear layer to embed grouped feature map.

    Args:
        input_resolution (tuple): The size of input patch resolution.
        in_channels (int): The num of input channels.
        expansion_ratio (Number): Expansion ratio of output channels. The num
            of output channels is equal to int(expansion_ratio * in_channels).
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Defaults to be equal with kernel_size.
        padding (int | tuple, optional): zero padding width in the unfold
            layer. Defaults to 0.
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Defaults to 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults to False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Defaults to dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 input_resolution,
                 in_channels,
                 expansion_ratio,
                 kernel_size=2,
                 stride=None,
                 padding=0,
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg)
        warnings.warn('The `PatchMerging` in mmcls will be deprecated. '
                      'Please use `mmcv.cnn.bricks.transformer.PatchMerging`. '
                      "It's more general and supports dynamic input shape")

        H, W = input_resolution
        self.input_resolution = input_resolution
        self.in_channels = in_channels
        self.out_channels = int(expansion_ratio * in_channels)

        if stride is None:
            stride = kernel_size
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self.sampler = nn.Unfold(kernel_size, dilation, padding, stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, self.out_channels, bias=bias)

        # See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        H_out = (H + 2 * padding[0] - dilation[0] *
                 (kernel_size[0] - 1) - 1) // stride[0] + 1
        W_out = (W + 2 * padding[1] - dilation[1] *
                 (kernel_size[1] - 1) - 1) // stride[1] + 1
        self.output_resolution = (H_out, W_out)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        x = self.sampler(x)  # B, 4*C, H/2*W/2
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)

        return x


def build_2d_sincos_position_embedding(patches_resolution,
                                       embed_dims,
                                       temperature=10000.,
                                       cls_token=False):
    """The function is to build position embedding for model to obtain the
    position information of the image patches."""

    if isinstance(patches_resolution, int):
        patches_resolution = (patches_resolution, patches_resolution)

    h, w = patches_resolution
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dims % 4 == 0, \
        'Embed dimension must be divisible by 4.'
    pos_dim = embed_dims // 4

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
        dim=1,
    )[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb


def build_1d_sincos_position_embedding(patches_size,
                                       embed_dims,
                                       temperature=10000.,
                                       cls_token=False):
    """The function is to build position embedding for model to obtain the
    position information of the sequence."""

    assert isinstance(patches_size, int)
    grid = torch.arange(patches_size, dtype=torch.float32)
    # grid = torch.meshgrid(grid)
    assert embed_dims % 2 == 0, \
        'Embed dimension must be divisible by 2 for 1D sequence.'
    pos_dim = embed_dims // 2

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out = torch.einsum('m,d->md', [grid.flatten(), omega])

    pos_emb = torch.cat([
        torch.sin(out),
        torch.cos(out)], dim=1)[None, :, :]

    if cls_token:
        cls_token_pe = torch.zeros([1, 1, embed_dims], dtype=torch.float32)
        pos_emb = torch.cat([cls_token_pe, pos_emb], dim=1)

    return pos_emb

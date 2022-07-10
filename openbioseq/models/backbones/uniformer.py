import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import NORM_LAYERS, Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmcv.cnn.utils.weight_init import constant_init, trunc_normal_init

from ..utils import MultiheadAttention, MultiheadAttentionWithRPE, PatchEmbed1d, to_2tuple
from ..registry import BACKBONES
from .base_backbone import BaseBackbone


@NORM_LAYERS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        # fix bug 'Grad strides do not match bucket view strides.' by contiguous()
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2).contiguous()


class MLP(nn.Module):
    """An implementation of vanilla FFN.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int): The hidden dimension of FFNs.
        out_features (int): The output dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMLP(nn.Module):
    """An implementation of Conv FFN in Uniformer.

    Args:
        in_features (int): The feature dimension.
        hidden_features (int): The hidden dimension of FFNs.
        out_features (int): The output dimension of FFNs.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        feature_Nd (str): Build Nd Conv in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.0,
                 feature_Nd="2d"):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if feature_Nd == "2d":
            _conv_layer = Conv2d
        elif feature_Nd == "1d":
            _conv_layer = nn.Conv1d
        self.fc1 = _conv_layer(in_features, hidden_features, 1)
        self.fc2 = _conv_layer(hidden_features, out_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    """Implement of Conv-based block in Uniformer.

    Args:
        embed_dims (int): The feature dimension.
        mlp_ratio (int): The hidden dimension for FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            spatial attention. Defaults to 5.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        feature_Nd (str): Build Nd Conv in {'1d', '2d'}. Default: '2d'.
        init_values (float): The init values of gamma. Defaults to 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 mlp_ratio=4.,
                 kernel_size=5,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                 init_values=1e-6,
                 feature_Nd="2d",
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        if feature_Nd == "2d":
            _conv_layer = Conv2d
            _token_size = (1, embed_dims, 1, 1)
            if "LN" in norm_cfg["type"]:
                norm_cfg["type"] = "LN2d"
        elif feature_Nd == "1d":
            _conv_layer = nn.Conv1d
            _token_size = (1, embed_dims, 1)
            if "BN" in norm_cfg["type"]:
                norm_cfg["type"] = "BN1d"

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # positional encoding
        self.pos_embed = _conv_layer(
            embed_dims, embed_dims, 3, padding=1, groups=embed_dims)

        # spatial attention
        self.conv1 = _conv_layer(embed_dims, embed_dims, 1)
        self.conv2 = _conv_layer(embed_dims, embed_dims, 1)
        self.attn = _conv_layer(embed_dims, embed_dims, kernel_size,
            padding=kernel_size // 2, groups=embed_dims)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # feed forward MLP
        self.ffn = ConvMLP(
            in_features=self.embed_dims,
            hidden_features=int(self.embed_dims * mlp_ratio),
            ffn_drop=drop_rate,
            act_cfg=act_cfg,
            feature_Nd=feature_Nd)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(_token_size), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(_token_size), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        x = x + self.pos_embed(x)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.conv2(self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class SABlock(nn.Module):
    """Implement of Self-attnetion-based Block in Uniformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        mlp_ratio (int): The hidden dimension for FFNs.
        window_size (int | None): Local window size of attention.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        qk_scale (int | None): Scale of the qk attention. Defaults to None.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        feature_Nd (str): Build Nd Conv in {'1d', '2d'}. Default: '2d'.
        init_values (float): The init values of gamma. Defaults to 0.0.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 mlp_ratio=4.,
                 window_size=None,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 feature_Nd="2d",
                 init_values=1e-6,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.feature_Nd = feature_Nd
        if feature_Nd == "2d":
            _conv_layer = Conv2d
        elif feature_Nd == "1d":
            _conv_layer = nn.Conv1d

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # positional encoding
        self.pos_embed = _conv_layer(
            embed_dims, embed_dims, 3, padding=1, groups=embed_dims)

        # self-attention
        if window_size is None:
            # attention without relative position bias
            self.attn = MultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)
        else:
            # attention with relative position bias
            self.attn = MultiheadAttentionWithRPE(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, proj_drop=drop_rate)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        # feed forward MLP
        self.ffn = MLP(
            in_features=embed_dims,
            hidden_features=int(embed_dims * mlp_ratio),
            ffn_drop=drop_rate,
            act_cfg=act_cfg)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((embed_dims)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (
                nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        x = x + self.pos_embed(x)
        if x.dim() == 4:
            B, N, H, W = x.shape
            x = x.flatten(2)
        x = x.transpose(1, 2)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        if self.feature_Nd == "2d":
            x = x.transpose(1, 2).reshape(B, N, H, W).contiguous()
        else:
            x = x.transpose(1, 2).contiguous()
        return x


class ConvEmbedding(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        out_features (int): The output dimension of FFNs.
        kernel_size (int): The conv kernel size of middle patch embedding.
            Defaults to 3.
        stride_size (int): The conv stride of middle patch embedding.
            Defaults to 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        feature_Nd (str): Build Nd Conv in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride_size=2,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN'),
                 feature_Nd="2d"):
        super(ConvEmbedding, self).__init__()

        if feature_Nd == "2d":
            _conv_layer = Conv2d
            if "LN" in norm_cfg["type"]:
                norm_cfg["type"] = "LN2d"
        elif feature_Nd == "1d":
            _conv_layer = nn.Conv1d
            if "BN" in norm_cfg["type"]:
                norm_cfg["type"] = "BN1d"

        self.projection = nn.Sequential(
            _conv_layer(in_channels, out_channels // 2, kernel_size=kernel_size,
                stride=stride_size, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            _conv_layer(out_channels // 2, out_channels, kernel_size=kernel_size,
                stride=stride_size, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class MiddleEmbedding(nn.Module):
    """An implementation of Conv middle embedding layer.

    Args:
        in_features (int): The feature dimension.
        out_features (int): The output dimension of FFNs.
        kernel_size (int): The conv kernel size of middle patch embedding.
            Defaults to 3.
        stride_size (int): The conv stride of middle patch embedding.
            Defaults to 2.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        feature_Nd (str): Build Nd Conv in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride_size=2,
                 norm_cfg=dict(type='BN'),
                 feature_Nd="2d"):
        super(MiddleEmbedding, self).__init__()

        if feature_Nd == "2d":
            _conv_layer = Conv2d
            if "LN" in norm_cfg["type"]:
                norm_cfg["type"] = "LN2d"
        elif feature_Nd == "1d":
            _conv_layer = nn.Conv1d
            if "BN" in norm_cfg["type"]:
                norm_cfg["type"] = "BN1d"

        self.projection = nn.Sequential(
            _conv_layer(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride_size, padding=kernel_size // 2),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class FixedPatchEmbed(nn.Module):
    """An implementation of Fixed Patch Embedding

    Args:
        input_size (int): The size or length of the input.
        patch_size (int): The stride of embedding conv.
        in_channels (int): The input feature dimension.
        out_features (int): The output dimension of embedding.
        feature_Nd (str): Build Nd Conv in {'1d', '2d'}. Default: '2d'.
    """
    def __init__(self,
                 input_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 feature_Nd="2d"):
        super().__init__()

        if feature_Nd == "2d":
            input_size = to_2tuple(input_size)
            patch_size = to_2tuple(patch_size)
            num_patches = (input_size[1] // patch_size[1]) * (input_size[0] // patch_size[0])
        else:
            num_patches = input_size
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if feature_Nd == "2d":
            self.norm = LayerNorm2d(embed_dims)
            self.proj = nn.Conv2d(
                in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)
        else:
            self.norm = nn.LayerNorm(embed_dims)
            self.proj = nn.Conv1d(
                in_channels, embed_dims, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if x.dim() == 4:
            _, _, H, W = x.shape
            assert H == self.input_size[0] and W == self.input_size[1], \
                f"Input image size ({H}*{W}) doesn't match model " \
                "({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)
            x = self.norm(x)
        elif x.dim() == 3:
            L = x.shape[-1]
            assert L == self.input_size, f"Input seq len {L} doesn't match the" \
                "model {self.input_size}."
            x = self.proj(x).transpose(1, 2)
            x = self.norm(x).transpose(1, 2).contiguous()

        return x


@BACKBONES.register_module()
class UniFormer(BaseBackbone):
    """Unified Transformer.

    A PyTorch implement of : `UniFormer: Unifying Convolution and Self-attention
    for Visual Recognition <https://arxiv.org/abs/2201.04676>`_

    Modified from the `official repo
    <https://github.com/Sense-X/UniFormer/tree/main/image_classification>`_

    Args:
        arch (str | dict): UniFormer architecture.
            If use string, choose from 'small' and 'base'.
            If use dict, it should have below keys:

            - **embed_dims** (List[int]): The dimensions of embedding.
            - **depths** (List[int]): The number of blocks in each stage.
            - **head_dim** (int): The dimensions of each head.
            - **patch_strides** (List[int]): The stride of each stage.
            - **conv_stem** (bool): Whether to use conv-stem.

            We provide UniFormer-Tiny (based on VAN-Tiny) in addition to the
            original paper. Defaults to 'small'.
        input_size (int | tuple): The expected input image or sequence shape.
            We don't support dynamic input shape, please set the argument to the
            true input shape. Defaults to 224.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to 3, means the last stage.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        conv_stem (bool): whether use overlapped patch stem.
        conv_kernel_size (int | list): The conv kernel size in the PatchEmbed.
            Defaults to 3, which is used when conv_stem=True.
        attn_kernel_size (int): The conv kernel size in the ConvBlock as the
            spatial attention. Defaults to 5.
        norm_cfg (dict): Config dict for self-attention normalization layer.
            Defaults to ``dict(type='LN')``.
        act_cfg (dict): The config dict for activation after each convolution.
            Defaults to ``dict(type='GELU')``.
        conv_norm_cfg (dict): Config dict for convolution normalization layer.
            Defaults to ``dict(type='BN')``.
        attention_types (str | list): Type of spatial attention in each stages.
            UniFormer uses ["Conv", "Conv", "MHSA", "MHSA"] by default.
        feature_Nd (str): Build Nd Conv in {'1d', '2d'}. Default: '2d'.
    """
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 160, 256],
                         'depths': [3, 4, 8, 3],
                         'head_dim': 32,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 4, 8, 3],
                         'head_dim': 64,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
        **dict.fromkeys(['s+', 'small_plus'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 5, 9, 3],
                         'head_dim': 32,
                         'patch_strides': [2, 2, 2, 2],
                         'conv_stem': True,
                        }),
        **dict.fromkeys(['s+_dim64', 'small_plus_dim64'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [3, 5, 9, 3],
                         'head_dim': 64,
                         'patch_strides': [2, 2, 2, 2],
                         'conv_stem': True,
                        }),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [5, 8, 20, 7],
                         'head_dim': 64,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [128, 192, 448, 640],
                         'depths': [5, 10, 24, 7],
                         'head_dim': 64,
                         'patch_strides': [4, 2, 2, 2],
                         'conv_stem': False,
                        }),
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 in_channels=3,
                 out_indices=(3,),
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 qk_scale=None,
                 init_values=1e-6,
                 conv_kernel_size=3,
                 attn_kernel_size=5,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 conv_norm_cfg=dict(type='BN'),
                 attention_types=["Conv", "Conv", "MHSA", "MHSA",],
                 final_norm=True,
                 frozen_stages=-1,
                 norm_eval=False,
                 feature_Nd="2d",
                 init_cfg=None,
                 **kwargs):
        super(UniFormer, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
            self.arch = arch.split("-")[0]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'head_dim', 'patch_strides', 'conv_stem'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch
            self.arch = 'small'

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.head_dim = self.arch_settings['head_dim']
        self.patch_strides = self.arch_settings['patch_strides']
        self.conv_stem = self.arch_settings['conv_stem']
        self.mlp_ratio = mlp_ratio
        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.feature_Nd = feature_Nd
        assert isinstance(out_indices, (int, tuple, list))
        if isinstance(out_indices, int):
            self.out_indices = [out_indices]

        self.attention_types = attention_types
        assert isinstance(attention_types, (str, list))
        if isinstance(attention_types, str):
            attention_types = [attention_types for i in range(self.num_stages)]
        assert len(attention_types) == self.num_stages
        assert isinstance(conv_kernel_size, (int, tuple, list))
        if isinstance(conv_kernel_size, int):
            conv_kernel_size = [conv_kernel_size for i in range(self.num_stages)]
        assert len(conv_kernel_size) == self.num_stages

        if "BN" in norm_cfg["type"]:
            norm_cfg["type"] = "BN1d"
        if "BN" in conv_norm_cfg["type"]:
            conv_norm_cfg["type"] = "BN2d" if feature_Nd == "2d" else "BN1d"
        elif "LN" in conv_norm_cfg["type"]:
            conv_norm_cfg["type"] = "LN2d" if feature_Nd == "2d" else "LN"

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        num_heads = [dim // self.head_dim for dim in self.embed_dims]

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            # build patch embedding
            if self.conv_stem:
                if i == 0:
                    patch_embed = ConvEmbedding(
                        in_channels=in_channels, out_channels=self.embed_dims[i],
                        kernel_size=conv_kernel_size[i], stride_size=self.patch_strides[i],
                        norm_cfg=conv_norm_cfg, act_cfg=act_cfg,
                        feature_Nd=feature_Nd)
                else:
                    patch_embed = MiddleEmbedding(
                        in_channels=self.embed_dims[i - 1], out_channels=self.embed_dims[i],
                        kernel_size=conv_kernel_size[i], stride_size=self.patch_strides[i],
                        norm_cfg=conv_norm_cfg, feature_Nd=feature_Nd)
            else:
                if self.feature_Nd == "2d":
                    patch_embed = PatchEmbed(
                        in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                        embed_dims=self.embed_dims[i],
                        kernel_size=self.patch_strides[i], stride=self.patch_strides[i],
                        padding=0 if self.patch_strides[i] % 2 == 0 else 'corner',
                        norm_cfg=norm_cfg,
                    )
                else:
                    patch_embed = PatchEmbed1d(
                        in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                        embed_dims=self.embed_dims[i],
                        kernel_size=self.patch_strides[i], stride=self.patch_strides[i],
                        padding=0 if self.patch_strides[i] % 2 == 0 else 'corner',
                        norm_cfg=norm_cfg,
                    )

            # build spatial mixing block
            if self.attention_types[i] == "Conv":
                blocks = nn.ModuleList([
                    ConvBlock(
                        embed_dims=self.embed_dims[i],
                        mlp_ratio=mlp_ratio,
                        kernel_size=attn_kernel_size,
                        drop_rate=drop_rate,
                        drop_path_rate=dpr[cur_block_idx + j],
                        norm_cfg=conv_norm_cfg,
                        feature_Nd=feature_Nd,
                        init_values=init_values,
                    ) for j in range(depth)
                ])
            elif self.attention_types[i] == "MHSA":
                blocks = nn.ModuleList([
                    SABlock(
                        embed_dims=self.embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratio,
                        window_size=None,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[cur_block_idx + j],
                        norm_cfg=norm_cfg,
                        feature_Nd=feature_Nd,
                        init_values=init_values,
                    ) for j in range(depth)
                ])
            else:
                raise NotImplementedError

            cur_block_idx += depth
            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)

        self.final_norm = final_norm
        if self.final_norm:
            for i in self.out_indices:
                if i < 0:
                    continue
                norm_layer = build_norm_layer(conv_norm_cfg, self.embed_dims[i])[1]
                self.add_module(f'norm{i}', norm_layer)

    def init_weights(self, pretrained=None):
        super(UniFormer, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            for m in self.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    trunc_normal_init(m, std=0.02, bias=0)
                elif isinstance(m, (
                    nn.LayerNorm, LayerNorm2d, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze blocks
            m = getattr(self, f'blocks{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            # freeze norm
            if i in self.out_indices and i > 0:
                if self.final_norm:
                    m = getattr(self, f'norm{i}')
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')

            x = patch_embed(x)
            if len(x) == 2:
                x, hw_shape = x  # patch_embed
                if self.feature_Nd == "2d":
                    x = x.reshape(x.shape[0],
                                *hw_shape, -1).permute(0, 3, 1, 2).contiguous()
                else:
                    x = x.transpose(1, 2).contiguous()

            if i == 0:
                x = self.drop_after_pos(x)
            for block in blocks:
                x = block(x)
            if i in self.out_indices:
                if self.final_norm:
                    norm_layer = getattr(self, f'norm{i}')
                    x = norm_layer(x)
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(UniFormer, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

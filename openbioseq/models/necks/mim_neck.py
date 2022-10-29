import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_norm_layer,
                      constant_init, trunc_normal_init)
from mmcv.runner.base_module import BaseModule
from openbioseq.models.backbones.vision_transformer import TransformerEncoderLayer

from .. import builder
from ..registry import NECKS
from ..utils import build_2d_sincos_position_embedding, trunc_normal_


@NECKS.register_module()
class BERTMLMNeck(BaseModule):
    """Pre-train Neck For BERT.

    This neck reconstructs the original image from the shrunk feature map.

    Args:
        in_channels (int): Channel dimension of the feature map.
        out_channels (int): Channel dimension of the output.
        encoder_stride (int): The total stride of the encoder.
        feature_Nd (str): Build Nd feature in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self,
                 in_channels=768,
                 out_channels=4,
                 encoder_stride=1,
                 feature_Nd="1d",
                 init_cfg=None):
        super(BERTMLMNeck, self).__init__(init_cfg)
        self.out_channels = out_channels
        self.encoder_stride = encoder_stride
        self.feature_Nd = feature_Nd
        assert feature_Nd == "1d"

        self.decoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, encoder_stride * out_channels),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (nn.LayerNorm)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        x = self.decoder(x)  # (BxL, C)
        if self.encoder_stride > 1:
            x = x.reshape(-1, self.out_channels)

        return x


@NECKS.register_module()
class MAEPretrainDecoder(BaseModule):
    """Decoder for MAE Pre-training.

    Args:
        num_patches (int): The number of total patches. Defaults to 196.
        patch_size (int): Image patch size. Defaults to 16.
        in_chans (int): The channel of input image. Defaults to 3.
        embed_dim (int): Encoder's embedding dimension. Defaults to 1024.
        decoder_embed_dim (int): Decoder's embedding dimension.
            Defaults to 512.
        decoder_depth (int): The depth of decoder. Defaults to 8.
        decoder_num_heads (int): Number of attention heads of decoder.
            Defaults to 16.
        mlp_ratio (int): Ratio of mlp hidden dim to decoder's embedding dim.
            Defaults to 4.
        norm_cfg (dict): Normalization layer. Defaults to LayerNorm.

    Some of the code is borrowed from
    `https://github.com/facebookresearch/mae`.

    Example:
        >>> from mmselfsup.models import MAEPretrainDecoder
        >>> import torch
        >>> self = MAEPretrainDecoder()
        >>> self.eval()
        >>> inputs = torch.rand(1, 50, 1024)
        >>> ids_restore = torch.arange(0, 196).unsqueeze(0)
        >>> level_outputs = self.forward(inputs, ids_restore)
        >>> print(tuple(level_outputs.shape))
        (1, 196, 768)
    """

    def __init__(self,
                 num_patches=196,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super(MAEPretrainDecoder, self).__init__(init_cfg)
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderLayer(
                decoder_embed_dim,
                decoder_num_heads,
                int(mlp_ratio * decoder_embed_dim),
                qkv_bias=True,
                norm_cfg=norm_cfg) for _ in range(decoder_depth)
        ])

        self.decoder_norm_name, decoder_norm = build_norm_layer(
            norm_cfg, decoder_embed_dim, postfix=1)
        self.add_module(self.decoder_norm_name, decoder_norm)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True)

    def init_weights(self):
        if self.init_cfg is not None:
            super(MAEPretrainDecoder, self).init_weights()
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)
        # initialize position embedding and mask token
        decoder_pos_embed = build_2d_sincos_position_embedding(
            int(self.num_patches**.5),
            self.decoder_pos_embed.shape[-1],
            cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float())
        trunc_normal_(self.mask_token, std=0.02)

    @property
    def decoder_norm(self):
        return getattr(self, self.decoder_norm_name)

    def forward(self, x, ids_restore):
        if isinstance(x, list):
            x = x[-1]
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


@NECKS.register_module()
class SimMIMNeck(BaseModule):
    """Pre-train Neck For SimMIM.

    This neck reconstructs the original image from the shrunk feature map.

    Args:
        in_channels (int): Channel dimension of the feature map.
        out_channels (int): Channel dimension of the output.
        encoder_stride (int): The total stride of the encoder.
        feature_Nd (str): Build Nd feature in {'1d', '2d'}. Default: '2d'.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=3,
                 encoder_stride=32,
                 feature_Nd="2d",
                 init_cfg=None):
        super(SimMIMNeck, self).__init__(init_cfg)
        self.out_channels = out_channels
        self.feature_Nd = feature_Nd
        assert feature_Nd in ["2d", "1d",]

        if feature_Nd == "2d":
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=encoder_stride**2 * out_channels,
                    kernel_size=1),
                nn.PixelShuffle(encoder_stride),
            )
        else:
            self.decoder = nn.Linear(
                in_channels, encoder_stride * out_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        x = self.decoder(x)
        if self.feature_Nd == "1d":
            x = x.reshape(-1, self.out_channels)  # (BxL, C)

        return x


@NECKS.register_module()
class NonLinearLMNeck(BaseModule):
    """Non-linear Neck For MIM/MLM Pre-training.

    Args:
        in_channels (int): Channel dimension of the feature map. It should
            be the decoder output channel if decoder_cfg is not None.
        in_chans (int): The channel of input image. Defaults to 3.
        encoder_stride (int): The total stride of the encoder.
        feature_Nd (str): Build Nd feature in {'1d', '2d'}. Default: '2d'.
        decoder_cfg (dict): Config dict for non-linear blocks. Defaults to None.
        act_cfg (dict): Whether to use an activation function. Defaults to None.
    """

    def __init__(self,
                 in_channels=128,
                 in_chans=3,
                 kernel_size=1,
                 encoder_stride=32,
                 feature_Nd="2d",
                 decoder_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super(NonLinearLMNeck, self).__init__(init_cfg)
        assert decoder_cfg is None or isinstance(decoder_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.decoder = builder.build_neck(decoder_cfg) \
            if decoder_cfg is not None else None
        self.activate = build_activation_layer(act_cfg) \
            if act_cfg is not None else None
        if feature_Nd == "2d":
            self.decoder_pred = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=encoder_stride**2 * in_chans,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                ),
                nn.PixelShuffle(encoder_stride),
            )
        else:
            self.decoder_pred = nn.Linear(
                in_channels, encoder_stride * in_chans)

    def init_weights(self):
        if self.init_cfg is not None:
            super(NonLinearLMNeck, self).init_weights()
            return
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                constant_init(m, val=1, bias=0)

    def forward(self, x):
        assert isinstance(x, list)
        if self.decoder is not None:
            dec = self.decoder([x[-1]])[0]
        else:
            dec = x[-1]

        dec = self.decoder_pred(dec)
        if self.activate is not None:
            dec = self.activate(dec)

        return [dec]

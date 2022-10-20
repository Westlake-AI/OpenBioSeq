import torch
import torch.nn as nn
from mmcv.cnn import (build_norm_layer,
                      constant_init, trunc_normal_init)
from mmcv.runner.base_module import BaseModule
from openbioseq.models.backbones.vision_transformer import TransformerEncoderLayer

from ..registry import NECKS
from ..utils import build_2d_sincos_position_embedding, trunc_normal_


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
            self.decoder = nn.Conv1d(
                in_channels=in_channels,
                out_channels=encoder_stride * out_channels,
                kernel_size=1,
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                trunc_normal_init(m, std=0.02, bias=0)

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        x = self.decoder(x)
        if self.feature_Nd == "1d":
            x = x.reshape(x.size(0), -1, self.out_channels)

        return x

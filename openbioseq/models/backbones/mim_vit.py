# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup mae_pretrain_vit.py
from openbioseq.models.utils.weight_init import trunc_normal_
import torch
from torch import nn
from mmcv.cnn import build_norm_layer, xavier_init, constant_init

from ..builder import BACKBONES
from .seq_transformer import SequenceTransformer
from .vision_transformer import VisionTransformer
from ..utils import build_2d_sincos_position_embedding


def forward_mae_masking(x, mask_ratio=0.75):
    """Generate the mask for MAE Pre-training.

    Args:
        x (torch.tensor): Image with data augmentation applied.
        mask_ratio (float): The mask ratio of total patches.
            Defaults to 0.75.

    Returns:
        tuple[Tensor, Tensor, Tensor]: masked input, mask and the ids
            to restore original data.

        - x_masked (Tensor): masked data.
        - mask (Tensor): mask used to mask data.
        - ids_restore (Tensor): ids to restore original data.
    """
    N, L, C = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(
        noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(
        x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


@BACKBONES.register_module()
class MAETransformer(SequenceTransformer):
    """Sequence Transformer for MAE pre-training.

    Args:
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
    """

    def __init__(self, mask_ratio=0.75, **kwargs):

        super().__init__(**kwargs)

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio

    def init_weights(self, pretrained=None):
        super(MAETransformer, self).init_weights(pretrained)

        if pretrained is None:
            # initialize position embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5), self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())
            w = self.patch_embed.projection.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            trunc_normal_(self.cls_token, std=0.02, bias=0)

            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_init(m, gain=1, bias=0, distribution='normal')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            constant_init(m, val=1, bias=0)

    def forward(self, x):
        """ MAE backbone only used for MAE model """
        B = x.shape[0]
        x, _ = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = forward_mae_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return (x, mask, ids_restore)


@BACKBONES.register_module()
class MAEViT(VisionTransformer):
    """Vision Transformer for MAE pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.75.
    """

    def __init__(self, mask_ratio=0.75, **kwargs):

        super().__init__(**kwargs)

        self.pos_embed.requires_grad = False
        self.mask_ratio = mask_ratio
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def init_weights(self, pretrained=None):
        super(MAEViT, self).init_weights(pretrained)

        if pretrained is None:
            # initialize position embedding in backbone
            pos_embed = build_2d_sincos_position_embedding(
                int(self.num_patches**.5), self.pos_embed.shape[-1],
                cls_token=True)
            self.pos_embed.data.copy_(pos_embed.float())
            w = self.patch_embed.projection.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            trunc_normal_(self.cls_token, std=0.02, bias=0)

            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            xavier_init(m, gain=1, bias=0, distribution='normal')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            constant_init(m, val=1, bias=0)

    def forward(self, x):
        """ MAE backbone only used for MAE model """
        B = x.shape[0]
        x, _ = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = forward_mae_masking(x, self.mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return (x, mask, ids_restore)


@BACKBONES.register_module()
class MIMVisionTransformer(VisionTransformer):
    """Vision Transformer for MIM-style model (Mask Image Modeling)
    classification (fine-tuning or linear probe).

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        data_size (int | tuple): Input data size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        finetune (bool): Whether or not do fine-tuning. Defaults to True.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='b',
                 data_size=224,
                 patch_size=16,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 finetune=True,
                 init_cfg=None):
        super().__init__(arch, data_size, patch_size, out_indices, drop_rate,
                         drop_path_rate, norm_cfg, final_norm,
                         output_cls_token, interpolate_mode, patch_cfg,
                         layer_cfgs, init_cfg)

        self.embed_dims = self.arch_settings['embed_dims']
        if not self.final_norm:
            _, self.fc_norm = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)

        self.finetune = finetune
        if not self.finetune:
            self._freeze_stages()

    def train(self, mode=True):
        super(MIMVisionTransformer, self).train(mode)
        if not self.finetune:
            self._freeze_stages()

    def _freeze_stages(self):
        """Freeze params in backbone when linear probing."""
        for _, param in self.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, _ = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)
        
        if not self.final_norm:
            x = x[:, 1:, :].mean(dim=1)
            outs = self.fc_norm(x)
        else:
            outs = x[:, 0]
        return [outs]


def simmim_random_masking(x, mask_ratio=0.65):
    N, L, _ = x.shape
    len_keep = int(L * (1 - mask_ratio))
    mask_idx = torch.randperm(L)[:len_keep].cuda()
    mask = torch.zeros((N, L)).cuda()
    mask[mask_idx] = 1
    return mask


def forward_simmim_masking(x, mask_token, mask=None, mask_mode=None):
    """ Apply the mask and mask_token for SimMIM Pre-training.

    Args:
        x (torch.tensor): Data with data augmentation applied.
        mask_token (torch.nn.Parameter): Learnable mask tokens.
        mask (torch.tensor): The mask of the input shape.
        mask_mode (str): Mode of masking methods.

    Returns:
        x_masked (Tensor): masked data.
    """
    if mask_mode is None:
        return x
    assert mask is not None
    B, L, _ = x.shape
    if mask_mode == 'mean':
        mask_token.data = x.mean(dim=[0, 1,], keepdim=True)
    mask = mask.flatten(1).unsqueeze(-1).type_as(x)  # (B, L, 1)

    if mask.size(1) + 1 == L:  # with cls_token
        mask_token = mask_token.expand(B, L-1, -1)
        x[:, 1:] = x[:, 1:] * (1. - mask) + mask_token * mask
    elif mask.size(1) == L:
        mask_token = mask_token.expand(B, L, -1)
        x = x * (1. - mask) + mask_token * mask
    else:
        raise NotImplementedError
    return x


@BACKBONES.register_module()
class SimMIMTransformer(SequenceTransformer):
    """Sequence Transformer for SimMIM pre-training.

    Args:
        mask_layer (int): Layer to start MIM (mask img and add mask_token).
            Defaults to 0.
        mask_token (str): Mode of applying mask token in {None, 'randn', 'zero',
            'learnable', 'mean'}. Defaults to 'learnable'.
    """

    def __init__(self,
                 mask_layer=0,
                 mask_ratio=0.65,
                 mask_token='learnable',
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_layer = mask_layer
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_token
        assert 0 <= self.mask_layer < self.num_layers
        assert self.mask_mode in [None, 'randn', 'zero', 'mean', 'learnable',]
        if self.mask_mode is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self, pretrained=None):
        super(SimMIMTransformer, self).init_weights(pretrained)

        if pretrained is None:
            # init mask token
            if self.mask_mode is not None:
                if self.mask_mode != 'zero':
                    trunc_normal_(self.mask_token, std=0.02, bias=0)
                if self.mask_mode != 'learnable':
                    self.mask_token.requires_grad = False

    def forward(self, x, mask=None):
        """Generate features for masked images.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor): Masks used to construct masked images.

        Returns:
            tuple: A tuple containing features from multi-stages.
        """
        x, seq_len = self.patch_embed(x)

        if self.mask_layer == 0:
            if mask is None:
                mask = simmim_random_masking(x, self.mask_ratio)
            x = forward_simmim_masking(
                x, self.mask_token, mask, self.mask_mode)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.resize_pos_embed(
            self.pos_embed,
            src_shape=self.patch_resolution,
            dst_shape=seq_len,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        for i, layer in enumerate(self.layers):
            if self.mask_layer == i+1:
                if mask is None:
                    mask = simmim_random_masking(x, self.mask_ratio)
                x = forward_simmim_masking(
                    x, self.mask_token, mask, self.mask_mode)

            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        if self.with_cls_token:
            x = x[:, 1:]

        return (x, mask)


@BACKBONES.register_module()
class SimMIMViT(VisionTransformer):
    """Vision Transformer for SimMIM pre-training.

    A PyTorch implement of: `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        mask_layer (int): Layer to start MIM (mask img and add mask_token).
            Defaults to 0.
        mask_token (str): Mode of applying mask token in {None, 'randn', 'zero',
            'learnable', 'mean'}. Defaults to 'learnable'.
    """

    def __init__(self,
                 mask_layer=0,
                 mask_token='learnable',
                 **kwargs):
        super().__init__(**kwargs)
        self.mask_layer = mask_layer
        self.mask_mode = mask_token
        assert 0 <= self.mask_layer < self.num_layers
        assert self.mask_mode in [None, 'randn', 'zero', 'mean', 'learnable',]
        if self.mask_mode is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

    def init_weights(self, pretrained=None):
        super(SimMIMViT, self).init_weights(pretrained)

        if pretrained is None:
            # init mask token
            if self.mask_mode is not None:
                if self.mask_mode != 'zero':
                    trunc_normal_(self.mask_token, std=0.02, bias=0)
                if self.mask_mode != 'learnable':
                    self.mask_token.requires_grad = False

    def forward(self, x, mask=None):
        """Generate features for masked images.

        Args:
            x (torch.Tensor): Input images.
            mask (torch.Tensor): Masks used to construct masked images.

        Returns:
            tuple: A tuple containing features from multi-stages.
        """
        x, _ = self.patch_embed(x)

        if self.mask_layer == 0:
            x = forward_simmim_masking(
                x, self.mask_token, mask, self.mask_mode)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            if self.mask_layer == i+1:
                x = forward_simmim_masking(
                    x, self.mask_token, mask, self.mask_mode)

            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                if self.with_cls_token:
                    x = x[:, 1:]
                B, L, C = x.shape
                H = W = int(L ** 0.5)
                x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
                outs.append(x)

        return outs

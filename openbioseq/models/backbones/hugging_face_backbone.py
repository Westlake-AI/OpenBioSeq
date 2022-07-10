try:
    import transformers
except ImportError:
    transformers = None

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_init

from openbioseq.utils import get_root_logger, print_log
from openbioseq.models.utils import resize_pos_embed
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def update_huggingface_config(config=None, config_args=dict()):
    """Update config of huggingface backbone.

    Args:
        config (dict | None): Config of huggingface backbone.
    """
    logger = get_root_logger()
    if config is None:
        logger.warning('This backbone does not have config')
        config = transformers.PretrainedConfig()
    config = config.from_dict(config_args)
    print_log(config, logger=logger)
    return config


@BACKBONES.register_module()
class HuggingFaceBackbone(BaseBackbone):
    """Wrapper to use backbones from HuggingFace library.

    More details can be found in
    `huggintface <https://github.com/rwightman/pytorch-image-models>`_.

    Args:
        model_name (str): Name of timm model to instantiate.
        in_channels (int): Number of input image channels. Defaults to 3.
        num_classes (int): Number of classes for classification head (used when
            features_only is False). Default to 1000.
        features_only (bool): Whether to extract feature pyramid (multi-scale
            feature maps from the deepest layer at each stride) by using timm
            supported `forward_features()`. Defaults to False.
        pretrained (bool): Whether to load pretrained weights.
            Defaults to False.
        checkpoint_path (str): Path of checkpoint to load at the last of
            ``timm.create_model``. Defaults to empty string, which means
            not loading.
        init_cfg (dict or list[dict], optional): Initialization config dict of
            OpenMMLab projects (removed!). Defaults to None.
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(self,
                 model_name,
                 pretrained=False,
                 config_args=dict(),
                 seq_len=128,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=None,
                 embed_update=False,
                 final_norm=True,
                 drop_rate=0.,
                 with_cls_token=True,
                 output_cls_token=True,
                 interpolate_mode='linear',
                 patch_cfg=dict(),
                 norm_cfg=dict(type='LN', eps=1e-12),
                 **kwargs):
        if transformers is None:
            raise RuntimeError(
                'Failed to import transformers. Please run "pip install transformers".')
        if not isinstance(pretrained, (bool, str)):
            raise TypeError('pretrained must be bool or str (pretrained name) in huggingface hub')

        super(HuggingFaceBackbone, self).__init__()
        arch_name = model_name + "Model"
        arch_conf = model_name + "Config"
        arch_conf = getattr(transformers, arch_conf)()
        arch_conf = update_huggingface_config(arch_conf, config_args)
        if embed_dims is not None:
            setattr(arch_conf, 'hidden_size', embed_dims)
            if getattr(arch_conf, 'intermediate_size') % embed_dims != 0:
                print_log(f"Warning, `intermediate_size` should be 4x{embed_dims}.")
                setattr(arch_conf, 'intermediate_size', int(4 * embed_dims))
        else:
            embed_dims = getattr(arch_conf, 'hidden_size')
        
        # update HuggingFace model
        hug_model = getattr(transformers, arch_name)(arch_conf)
        if pretrained:
            hug_model = hug_model.from_pretrained(pretrained)
        hug_model.post_init()  # init weights
        self.embeddings = hug_model.embeddings if not embed_update else None
        self.encoder = hug_model.encoder

        # update embedding
        self.embed_update = embed_update
        self.embed_dims = embed_dims
        if embed_update:
            patch_cfg['type'] = patch_cfg.get('type', 'PatchEmbed1d')
            assert patch_cfg['type'] in ['PatchEmbed', 'PatchEmbed1d',]
            _patch_cfg = dict(
                in_channels=in_channels,
                input_size=seq_len,
                embed_dims=self.embed_dims,
                conv_type='Conv1d',
                kernel_size=patch_cfg.get('kernel_size', patch_size),
                stride=patch_cfg.get('kernel_size', patch_size),
            )
            self.embeddings = eval(patch_cfg['type'])(**_patch_cfg)
            self.num_patches = self.embeddings.init_out_size

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
        
        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self, pretrained=None):
        super(HuggingFaceBackbone, self).init_weights(pretrained)

        if pretrained is None:
            if self.init_cfg is not None:
                return
            if self.embed_update:
                for m in self.embeddings.modules():
                    if isinstance(m, (nn.Conv1d, nn.Linear)):
                        trunc_normal_init(m, std=0.02, bias=0)
                # pos_embed & cls_token
                nn.init.trunc_normal_(self.pos_embed, mean=0, std=.02)
                if self.cls_token is not None:
                    nn.init.trunc_normal_(self.cls_token, mean=0, std=.02)

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
            pos_embed_shape = self.embeddings.init_out_size

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
        # embedding
        if self.embed_update:
            B = x.shape[0]
            x, seq_len = self.embeddings(x)
            
            if self.cls_token is not None:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
            x = x + resize_pos_embed(
                self.pos_embed,
                src_shape=self.num_patches,
                dst_shape=seq_len,
                mode=self.interpolate_mode,
                num_extra_tokens=self.num_extra_tokens)
            x = self.drop_after_pos(x)
        else:
            if self.embeddings is not None:  # original embed
                x = self.embeddings(x)
        # encoder
        x = self.encoder(x)[0]  # 'last_hidden_state'

        # final norm
        x = self.norm1(x)
        B, L, C = x.shape
        if self.with_cls_token:
            patch_token = x[:, 1:].reshape(B, L-1, C)
            patch_token = patch_token.permute(0, 2, 1)
            cls_token = x[:, 0]
        else:
            patch_token = x.permute(0, 2, 1)  # (B, C, L)
            cls_token = None
        if self.output_cls_token:
            out = [patch_token, cls_token]
        else:
            out = patch_token
        
        return [out]

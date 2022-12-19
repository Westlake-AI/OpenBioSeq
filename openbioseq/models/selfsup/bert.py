import random
import torch

from openbioseq.utils import print_log
from ..classifiers import BaseModel
from ..utils import accuracy, AdaptivePadding1d
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class BERT(BaseModel):
    """BERT.

    Implementation of `BERT: Pre-training of Deep Bidirectional Transformers
    for Language Understanding <https://arxiv.org/abs/1810.04805>`_.
    
    Args:
        backbone (dict): Config dict for encoder.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        mask_ratio (float): Masking ratio for MLM pre-training. Default to 0.15.
        pretrained (str, optional): Path to pre-trained weights. Default to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 mask_ratio=0.15,
                 spin_stride=[],
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(BERT, self).__init__(init_cfg, **kwargs)
        assert isinstance(backbone, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.mask_ratio = mask_ratio
        self.spin_stride = list() if not isinstance(spin_stride, (tuple, list)) \
            else list(spin_stride)

        self.patch_size = getattr(self.backbone, 'patch_size', 1)
        if self.patch_size > 1:
            self.padding = AdaptivePadding1d(
                kernel_size=self.patch_size, stride=self.patch_size,
                dilation=1, padding="corner")
        else:
            self.padding = None

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(BERT, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()

    def forward_backbone(self, data):
        """Forward backbone.

        Args:
            data (Tensor): Input images of shape (N, C, H, W) or (N, C, L).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs of (N, C).
        """
        x = self.backbone(data)
        if len(x) == 3:
            # return cls_token, yeilding better performances than patch tokens
            return [x[0][:, 0]]
        elif len(x) == 2:
            return [x[0][-1]]  # return cls_token
        else:
            return x

    def forward_train(self, data, **kwargs):
        """Forward computation during training.

        Args:
            data (Tensor): Input images of shape (N, C, L).
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert data.dim() == 3, "data shape should be (N, C, L)"
        if self.padding is not None:
            data = self.padding(data)
        B, _, L = data.size()

        if len(self.spin_stride) > 0:
            spin = random.choices(self.spin_stride, k=1)[0]
            assert L % spin == 0 and spin >= 1
            spin_L = L // spin
            mask = torch.bernoulli(torch.full([1, spin_L], self.mask_ratio)).cuda()
            mask = mask.view(1, spin_L, 1).expand(1, spin_L, spin).reshape(1, L)
        else:
            mask = torch.bernoulli(torch.full([1, L], self.mask_ratio)).cuda()
        latent, _ = self.backbone(data, mask=None)
        latent = latent.reshape(-1, latent.size(2))  # (B, L, C) -> (BxL, C)
        data_rec = self.neck(latent)
        if isinstance(data_rec, list):
            data_rec = data_rec[-1]

        target = data.permute(0, 2, 1).reshape(-1, data.size(1))  # (B, C, L) -> (BxL, C)
        if data_rec.dim() == 3:
            data_rec = data_rec.reshape(-1, data_rec.size(2))  # (B, L, C) -> (BxL, C)
        mask = mask.view(1, L).expand(B, L).reshape(-1, 1)
        losses = self.head(target, data_rec, mask)

        mask = mask.squeeze().bool()
        losses['acc'] = accuracy(data_rec[mask], target[mask])

        return losses

# reference: https://github.com/open-mmlab/mmselfsup/tree/master/mmselfsup/models/algorithms
# modified from mmselfsup mae.py
from openbioseq.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    
    Args:
        backbone (dict): Config dict for encoder.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(MAE, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(MAE, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()

    def forward_backbone(self, data):
        """Forward backbone.

        Args:
            data (Tensor): Input images of shape (N, C, H, W) or (N, C, D).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs of (N, D).
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
            data (Tensor): Input images of shape (N, C, H, W) or (N, C, D).
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        latent, mask, ids_restore = self.backbone(data)
        pred = self.neck(latent, ids_restore)
        losses = self.head(data, pred, mask)

        return losses

from openbioseq.utils import print_log

from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class SimMIM(BaseModel):
    """SimMIM.

    Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
    <https://arxiv.org/abs/2111.09886>`_.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        save (bool): Saving reconstruction results. Defaults to False.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(SimMIM, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(SimMIM, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()

    def forward_backbone(self, data):
        """Forward backbone.

        Args:
            data (list[Tensor]): A list of input images with shape
                (N, C, H, W) or (N, C, L). Typically these should be mean
                centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs of (N, C).
        """
        x = self.backbone(data)
        if len(x) == 3:
            # return cls_token, yeilding better performances than patch tokens
            x = x[0][:, 0]
        else:
            x = x[0][-1]  # return cls_token
        return [x]

    def forward_train(self, data, **kwargs):
        """Forward the masked image and get the reconstruction loss.

        Args:
            x (List[torch.Tensor, torch.Tensor]): Data and masks.

        Returns:
            dict: Reconstructed loss.
        """
        mask = kwargs.get('mask', None)
        if isinstance(data, list):
            data, mask = data
        if isinstance(mask, list):
            mask, _ = mask

        x_latent = self.backbone(data, mask)
        x_rec = self.neck(x_latent[0])
        if isinstance(x_rec, list):
            x_rec = x_rec[-1]
        losses = self.head(data, x_rec, mask)

        return losses

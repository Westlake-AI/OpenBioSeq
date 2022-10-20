import torch

from openbioseq.utils import print_log
from ..classifiers import BaseModel
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
        super(BERT, self).__init__(init_cfg, **kwargs)
        assert isinstance(backbone, dict) and isinstance(head, dict)
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

        latent, mask = self.backbone(data, mask=None)
        data_rec = self.neck(latent[0])
        if isinstance(data_rec, list):
            data_rec = data_rec[-1]
        
        ####
        print("BERT:", data.shape, data_rec.shape)
        ####
        losses = self.head(
            data.reshape(-1, data.size(2)),
            data_rec.reshape(-1, data.size(2)),
            mask.reshape(-1, 1))

        return losses

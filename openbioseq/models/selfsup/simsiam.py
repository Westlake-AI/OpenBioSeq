import torch.nn as nn

from openbioseq.utils import print_log
from ..classifiers import BaseModel
from .. import builder
from ..registry import MODELS


@MODELS.register_module
class SimSiam(BaseModel):
    """SimSiam.

    Implementation of `Exploring Simple Siamese Representation Learning
    <https://arxiv.org/abs/2011.10566>`_.
    The operation of fixing learning rate of predictor is in
    `core/hooks/simsiam_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions. Defaults to None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(SimSiam, self).__init__(init_cfg, **kwargs)
        assert isinstance(neck, dict) and isinstance(head, dict)
        self.encoder = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.encoder[0]
        self.neck = self.encoder[1]
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(SimSiam, self).init_weights()

        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder[0].init_weights(pretrained=pretrained)  # backbone
        self.encoder[1].init_weights(init_linear='kaiming')  # projection
        # init the predictor in the head
        self.head.init_weights(init_linear='normal')

    def forward_train(self, data, **kwargs):
        """Forward computation during training.

        Args:
            data (list[Tensor]): A list of input images with shape (N, C, H, W)
                or (N, C, L). Typically these should be mean centered
                and std scaled.
        Returns:
            loss[str, Tensor]: A dictionary of loss components
        """
        assert isinstance(data, list) and len(data) >= 2
        data_v1 = data[0].contiguous()
        data_v2 = data[1].contiguous()

        z1 = self.encoder(data_v1)[0]  # NxC
        z2 = self.encoder(data_v2)[0]  # NxC

        losses = 0.5 * (self.head(z1, z2)['loss'] + self.head(z2, z1)['loss'])
        return dict(loss=losses)

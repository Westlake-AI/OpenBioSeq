import copy
import torch
from openbioseq.utils import print_log

from .base_model import BaseModel
from .. import builder
from ..registry import MODELS
from ..utils import PlotTensor


@MODELS.register_module
class Classification(BaseModel):
    """Simple supervised classification or regression.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 save_name="cls",
                 init_cfg=None,
                 **kwargs):
        super(Classification, self).__init__(init_cfg, **kwargs)
        self.backbone = builder.build_backbone(backbone)
        assert isinstance(head, dict)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        # save plots
        self.save_name = save_name
        self.save = False
        self.ploter = PlotTensor(data_type="seq", apply_inv=False)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            self.neck.init_weights(init_linear='kaiming')
        self.head.init_weights()

    def forward_train(self, data, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            data (Tensor): Input images of shape (N, C, D) or (N, C, H, W).
                Typically these should be mean centered and std scaled.
            gt_label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.backbone(data)
        if self.save:
            self.plot_latent(data, copy.copy(x), target=gt_label)
        
        x = [x[-1]]
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        losses = self.head.loss(outs, gt_label.squeeze())
        
        return losses

    def forward_test(self, data, **kwargs):
        x = self.backbone(data)  # tuple
        x = [x[-1]]
        if self.with_neck:
            x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))
    
    def plot_latent(self, data, latent, target=None, mode="channel-mean"):
        """ visualize latent space """

        nrow = 4 if target is None else int(target.max().detach().cpu().numpy())
        # nrow = 100
        if isinstance(latent, torch.Tensor):
            latent = [latent]

        if mode == "channel-wise":
            data_max, _ = torch.max(data, dim=1, keepdim=True)
            data_min, _ = torch.min(data, dim=1, keepdim=True)
            data = (data - data_min) / (data_max - data_min + 1e-10)
            for i in range(len(latent)):
                latent_max, _ = torch.max(latent[i], dim=1, keepdim=True)
                latent_min, _ = torch.min(latent[i], dim=1, keepdim=True)
                latent[i] = (latent[i] - latent_min) / (latent_max - latent_min + 1e-10)
        elif mode == "channel-mean":
            data = data.mean(dim=1, keepdim=True)
            for i in range(len(latent)):
                latent[i] = latent[i].mean(dim=1, keepdim=True)
        
        data = data[:nrow]
        for i in range(len(latent)):
            latent[i] = latent[i][:nrow]
        if target is not None:
            target = target[:nrow].detach().cpu().numpy()

        data_dict = dict(data=dict())
        if mode == "channel-wise":
            for c in range(data.size(1)):
                data_dict['data']["channel_"+str(c)] = data[0, c, ...].detach().cpu().numpy()
            for i in range(len(latent)):
                data_dict["latent_space_"+str(i)] = dict()
                for c in range(latent[i].size(1)):
                    data_dict["latent_space_"+str(i)]["channel_"+str(c)] = \
                        latent[i][0, c, ...].detach().cpu().numpy()
        elif mode == "channel-mean":
            if target is not None:
                for i in range(data.size(0)):
                    data_dict['data'][str(target[i])] = data[i, 0, ...].detach().cpu().numpy()
            else:
                data_dict['data']["channel_mean"] = data[0, 0, ...].detach().cpu().numpy()
            for i in range(len(latent)):
                data_dict["latent_space_"+str(i)] = dict()
                if target is not None:
                    for j in range(data.size(0)):
                        data_dict["latent_space_"+str(i)][str(target[j])] = \
                            latent[i][j, 0, ...].detach().cpu().numpy()
                else:
                    data_dict["latent_space_"+str(i)]["channel_mean"] = \
                        latent[i][0, 0, ...].detach().cpu().numpy()
        
        assert self.save_name.endswith(".png")
        self.ploter.plot(data=data_dict, save_name=self.save_name)

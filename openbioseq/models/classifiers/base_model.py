import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from mmcv.runner import BaseModule, auto_fp16

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class BaseModel(BaseModule, metaclass=ABCMeta):
    """Base model class for supervised, semi- and self-supervised learning."""

    def __init__(self,
                 init_cfg=None,
                 use_transformer=False,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.fp16_enabled = False
        self.use_transformer = use_transformer
        self._is_init = False
        if init_cfg is not None:
            self.init_cfg = init_cfg
        self.backbone = nn.Identity()
        self.neck = None
        self.head = nn.Identity()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        if not self._is_init:
            super(BaseModel, self).init_weights()
        else:
            warnings.warn('This module has bee initialized, \
                please call initialize(module, init_cfg) to reinitialize it')

    def forward_backbone(self, data):
        """Forward backbone.

        Args:
            data (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(data)
        return x

    @abstractmethod
    def forward_train(self, data, **kwargs):
        """
        Args:
            data ([Tensor): List of tensors. Typically these should be
                mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    def forward_test(self, data, **kwargs):
        """
        Args:
            data (Tensor): List of tensors. Typically these should be
                mean centered and std scaled.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        pass

    def forward_vis(self, data, **kwargs):
        """Forward backbone features for visualization.

        Args:
            data (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        feat = self.forward_backbone(data)  # tuple
        if self.use_transformer:
            if len(feat) == 3:
                feat = feat[0][:, 0]  # return cls_token
            elif len(feat) == 2:
                feat = feat[0][-1]  # return cls_token
            else:
                feat = feat[-1]
            if isinstance(feat, list):
                feat = feat[-1]  # return cls_token
        else:
            feat = feat[-1]
        
        # apply pooling
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, 1)  # NxCx1x1
        elif feat.dim() == 3:
            feat = F.adaptive_avg_pool1d(feat, 1)  # NxCx1
        keys = ['gap']
        outs = [feat.view(feat.size(0), -1).cpu()]
        return dict(zip(keys, outs))
    
    def forward_calibration(self, data, **kwargs):
        x = self.backbone(data)
        preds = self.head(x)
        return preds

    @auto_fp16(apply_to=('data', ))
    def forward(self, data, mode='train', **kwargs):
        """Forward function of model.

        Calls either forward_train, forward_test or forward_backbone function
        according to the mode.
        """
        if mode == 'train':
            return self.forward_train(data, **kwargs)
        elif mode == 'test':
            return self.forward_test(data, **kwargs)
        elif mode == 'calibration':
            return self.forward_calibration(data, **kwargs)
        elif mode == 'extract':
            if len(data.size()) > 4:
                data = data[:, 0, ...].contiguous()  # contrastive data
            return self.forward_backbone(data)
        elif mode == 'vis':
            return self.forward_vis(data, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer=None):
        """The iteration step during training.

        *** replacing `batch_processor` in `EpochBasedRunner` in old version ***

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: Dict of outputs. The following fields are contained.
                - loss (torch.Tensor): A tensor for back propagation, which \
                    can be a weighted sum of multiple losses.
                - log_vars (dict): Dict contains all the variables to be sent \
                    to the logger.
                - num_samples (int): Indicates the batch size (when the model \
                    is DDP, it means the batch size on each GPU), which is \
                    used for averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['data'], list):
            num_samples = len(data['data'][0].data)
        else:
            num_samples = len(data['data'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if isinstance(data['img'], list):
            num_samples = len(data['img'][0].data)
        else:
            num_samples = len(data['img'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

        return outputs

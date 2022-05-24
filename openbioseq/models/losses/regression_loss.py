import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


@LOSSES.register_module()
class RegressionLoss(nn.Module):
    r"""Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        use_mix_decouple (bool): Whether to use decoupled mixup version of
            CrossEntropyLoss with the 'soft' CE implementation. Default to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
    """

    def __init__(self,
                 mode="mse",
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(RegressionLoss, self).__init__()
        self.mode = mode
        self.reduction = reduction
        self.loss_weight = loss_weight
        assert mode in ["mse_loss", "l1_loss", "smooth_l1_loss",]
        # loss func
        self.criterion = getattr(F, self.mode)
    
    def forward(self,
                pred,
                label,
                reduction_override=None,
                **kwargs):
        r"""caculate loss
        
        Args:
            pred (tensor): Predicted logits of (N,).
            label (tensor): Groundtruth label of (N,).
        """
        assert reduction_override in (None, 'none', 'mean',)
        reduction = (
            reduction_override if reduction_override is not None else self.reduction)        
        loss_reg = self.loss_weight * self.criterion(
            pred, label, reduction=reduction, **kwargs)
        return loss_reg

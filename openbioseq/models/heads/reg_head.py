import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, normal_init

from ..utils import regression_error, trunc_normal_init
from ..registry import HEADS
from ..builder import build_loss


@HEADS.register_module
class RegHead(nn.Module):
    """Simplest regression head, with only one fc layer.
    
    Args:
        with_avg_pool (bool): Whether to use GAP before this head.
        loss (dict): Config of classification loss.
        in_channels (int): Number of channels in the input feature map.
        frozen (bool): Whether to freeze the parameters.
    """

    def __init__(self,
                 with_avg_pool=False,
                 loss=dict(type='RegressionLoss', loss_weight=1.0, mode="mse_loss"),
                 in_channels=2048,
                 out_channels=1,
                 frozen=False):
        super(RegHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.out_channels = out_channels

        # loss
        if loss is not None:
            assert isinstance(loss, dict)
            self.criterion = build_loss(loss)
        else:
            loss = dict(type='RegressionLoss', loss_weight=1.0, mode="mse_loss")
            self.criterion = build_loss(loss)
        # fc layer
        self.fc = nn.Linear(in_channels, out_channels)
        if frozen:
            self.frozen()

    def frozen(self):
        self.fc.eval()
        for param in self.fc.parameters():
            param.requires_grad = False

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming', 'trunc_normal'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                elif init_linear == 'kaiming':
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif init_linear == 'trunc_normal':
                    trunc_normal_init(m, std=std, bias=bias)

    def forward(self, x):
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        x = self.fc(x).squeeze()
        return [x]

    def loss(self, score, labels, **kwargs):
        """" regression loss forward
        
        Args:
            score (list): Score should be [tensor] in (N,).
            labels (tuple or tensor): Labels should be tensor (N,) by default.
        """
        losses = dict()
        assert isinstance(score, (tuple, list)) and len(score) == 1
        
        # computing loss
        labels = labels.type_as(score[0])
        losses['loss'] = self.criterion(score[0], labels, **kwargs)
        # compute error
        losses['mse'], _ = regression_error(score[0], labels, average_mode='mean')
        
        return losses

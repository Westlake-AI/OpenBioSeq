import random
import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, constant_init

from ..registry import BACKBONES
from .base_backbone import BaseBackbone
from ..utils import grad_batch_shuffle_ddp, grad_batch_unshuffle_ddp


class BasicBlock(nn.Module):
    """BasicBlock for Wide ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        stride (int): stride of the block. Default: 1.
        drop_rate (float): Dropout ratio in the residual block. Default: 0.
        feature_Nd (str): Build Nd network in {'1d', '2d'}. Default: '2d'.
        activate_before_residual (bool): Since the first conv in WRN doesn't
            have bn-relu behind, we use the bn1 and relu1 in the block1 to
            make up the ``conv1-bn1-relu1`` structure. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 kernel_size=3,
                 drop_rate=0.0,
                 feature_Nd="2d",
                 activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.equalInOut = (in_channels == out_channels)
        self.convShortcut = None
        if feature_Nd == "1d":
            self.bn1 = nn.BatchNorm1d(in_channels, momentum=0.001, eps=0.001)
            self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=kernel_size // 2, bias=True)
            self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.001, eps=0.001)
            self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                            stride=1, padding=kernel_size // 2, bias=True)
            if stride != 1 or not self.equalInOut:
                self.convShortcut = nn.Conv1d(in_channels, out_channels,
                                        kernel_size=1, stride=stride, padding=0, bias=True)
        elif feature_Nd == "2d":
            self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.001, eps=0.001)
            self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            stride=stride, padding=kernel_size // 2, bias=True)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)
            self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                            stride=1, padding=kernel_size // 2, bias=True)
            if stride != 1 or not self.equalInOut:
                self.convShortcut = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=1, stride=stride, padding=0, bias=True)
        else:
            raise NotImplementedError
        self.dropout = nn.Dropout(float(drop_rate)) \
            if float(drop_rate) > 0 else nn.Identity()
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(x)))
        else:
            out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out)))
        out = self.dropout(out)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """" Network Block (stage) in Wide ResNet """
    
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 block,
                 stride,
                 kernel_size=3,
                 drop_rate=0.0,
                 feature_Nd="2d",
                 activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(int(num_layers)):
            layers.append(block(i == 0 and in_channels or out_channels, out_channels,
                                i == 0 and stride or 1, kernel_size, drop_rate,
                                feature_Nd, activate_before_residual))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


@BACKBONES.register_module()
class WideResNet(BaseBackbone):
    """Wide Residual Networks backbone.

    Please refer to the `paper <https://arxiv.org/pdf/1605.07146.pdf>`_ for
    details.
        https://github.com/szagoruyko/wide-residual-networks

    Args:
        first_kernel (int): Kernel size of the first conv. Default: 3.
        first_stride (int): Stride of the first conv. Default: 1.
        block_stride (int): Stride of the first block. Default: 1.
        in_channels (int): Number of input image channels. Default: 3.
        depth (int): Network depth, from {10, 28, 37}, total 3 stages.
        widen_factor (int): Width of each stage convolution block. Default: 2.
        kernel_size (int): Conv kernel size in each block. Default: 3.
        drop_rate (float): Dropout ratio in residual blocks. Default: 0.
        out_indices (Sequence[int]): Output from which stages. If only one
            stage is specified, a single tensor (feature map) is returned,
            otherwise multiple stages are specified, a tuple of tensors will
            be returned. Default: ``(2, )``.
        feature_Nd (str): Build Nd network in {'1d', '2d'}. Default: '2d'.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
    """

    def __init__(self,
                 first_kernel=3,
                 first_stride=1,
                 block_stride=1,
                 in_channels=3,
                 depth=28,
                 widen_factor=2,
                 kernel_size=3,
                 drop_rate=0.0,
                 out_indices=(0, 1, 2,),
                 feature_Nd="2d",
                 frozen_stages=-1,
                 norm_eval=False):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        assert feature_Nd in ["1d", "2d"]

        # 1st conv before any network block, e.g., 3x3
        if feature_Nd == "1d":
            self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size=first_kernel,
                                   stride=first_stride, padding=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=first_kernel,
                                   stride=first_stride, padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], BasicBlock, stride=block_stride,
            kernel_size=kernel_size, drop_rate=drop_rate, feature_Nd=feature_Nd,
            activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], BasicBlock, stride=2,
            kernel_size=kernel_size, drop_rate=drop_rate, feature_Nd=feature_Nd)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], BasicBlock, stride=2,
            kernel_size=kernel_size, drop_rate=drop_rate, feature_Nd=feature_Nd)
        
        # original: global average pooling and classifier (in head)
        if feature_Nd == "1d":
            self.bn1 = nn.BatchNorm1d(channels[3], momentum=0.001, eps=0.001)
        else:
            self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.channels = channels[3]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        
        self._freeze_stages()
    
    def init_weights(self, pretrained=None):
        super(WideResNet, self).init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1]:
                for param in m.parameters():
                    param.requires_grad = False
            for i in range(self.frozen_stages + 1):
                m = getattr(self, 'block{}'.format(i+1))
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
            if self.frozen_stages == 2:
                for m in [self.bn1]:
                    for param in m.parameters():
                        param.requires_grad = False

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        for i in range(3):
            block_i = getattr(self, 'block{}'.format(i+1))
            x = block_i(x)
            if i == 2:  # after block3
                x = self.relu(self.bn1(x))
            if i in self.out_indices:
                outs.append(x)
                if len(self.out_indices) == 1:
                    return tuple(outs)
        return tuple(outs)

    def train(self, mode=True):
        super(WideResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()


@BACKBONES.register_module()
class WideResNet_Mix(WideResNet):
    """Wide-ResNet Support ManifoldMix and its variants

    Provide a port to mixup the latent space.
    """

    def __init__(self, **kwargs):
        super(WideResNet_Mix, self).__init__(**kwargs)
    
    def _feature_mixup(self, x, mask, dist_shuffle=False, idx_shuffle_mix=None, cross_view=False,
            BN_shuffle=False, idx_shuffle_BN=None, idx_unshuffle_BN=None, **kwargs):
        """ mixup two feature maps with the pixel-wise mask
        
        Args:
            x, mask (tensor): Input x [N,C,H,W] and mixup mask [N, \*, H, W].
            dist_shuffle (bool): Whether to shuffle cross gpus.
            idx_shuffle_mix (tensor): Shuffle indice of [N,1] to generate x_.
            cross_view (bool): Whether to view the input x as two views [2N, C, H, W],
                which is usually adopted in self-supervised and semi-supervised settings.
            BN_shuffle (bool): Whether to do shuffle cross gpus for shuffle_BN.
            idx_shuffle_BN (tensor): Shuffle indice to utilize shuffle_BN cross gpus.
            idx_unshuffle_BN (tensor): Unshuffle indice for the shuffle_BN (in pair).
        """
        # adjust mixup mask
        assert mask.dim() == 4 and mask.size(1) <= 2
        if mask.size(1) == 1:
            mask = [mask, 1 - mask]
        else:
            mask = [
                mask[:, 0, :, :].unsqueeze(1), mask[:, 1, :, :].unsqueeze(1)]
        # undo shuffle_BN for ssl mixup
        if BN_shuffle:
            assert idx_unshuffle_BN is not None and idx_shuffle_BN is not None
            x = grad_batch_unshuffle_ddp(x, idx_unshuffle_BN)  # 2N index if cross_view
        
        # shuffle input
        if dist_shuffle==True:  # cross gpus shuffle
            assert idx_shuffle_mix is not None
            if cross_view:
                N = x.size(0) // 2
                detach_p = random.random()
                x_ = x[N:, ...].clone().detach() if detach_p < 0.5 else x[N:, ...]
                x = x[:N, ...] if detach_p < 0.5 else x[:N, ...].detach()
                x_, _, _ = grad_batch_shuffle_ddp(x_, idx_shuffle_mix)
            else:
                x_, _, _ = grad_batch_shuffle_ddp(x, idx_shuffle_mix)
        else:  # within each gpu
            if cross_view:
                # default: the input image is shuffled
                N = x.size(0) // 2
                detach_p = random.random()
                x_ = x[N:, ...].clone().detach() if detach_p < 0.5 else x[N:, ...]
                x = x[:N, ...] if detach_p < 0.5 else x[:N, ...].detach()
            else:
                x_ = x[idx_shuffle_mix, :]
        assert x.size(3) == mask[0].size(3), \
            "mismatching mask x={}, mask={}.".format(x.size(), mask[0].size())
        mix = x * mask[0] + x_ * mask[1]

        # redo shuffle_BN for ssl mixup
        if BN_shuffle:
            mix, _, _ = grad_batch_shuffle_ddp(mix, idx_shuffle_BN)  # N index
        
        return mix

    def forward(self, x, mix_args=None):
        """ only support mask-based mixup policy """
        # latent space mixup
        if mix_args is not None:
            assert isinstance(mix_args, dict)
            mix_layer = mix_args["layer"]  # {0, 1, 2,}
            if mix_args["BN_shuffle"]:
                x, _, idx_unshuffle = grad_batch_shuffle_ddp(x)  # 2N index if cross_view
            else:
                idx_unshuffle = None
        else:
            mix_layer = -1
        
        # input mixup
        if mix_layer == 0:
            x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)
        # normal conv1
        x = self.conv1(x)

        outs = []
        # block 1 to 3
        for i in range(3):
            block_i = getattr(self, 'block{}'.format(i+1))
            x = block_i(x)
            if i == 2:  # after block3
                x = self.relu(self.bn1(x))
            if i in self.out_indices:
                outs.append(x)
                if len(self.out_indices) == 1:
                    return tuple(outs)
            if i+1 == mix_layer:
                x = self._feature_mixup(x, idx_unshuffle_BN=idx_unshuffle, **mix_args)
        return tuple(outs)

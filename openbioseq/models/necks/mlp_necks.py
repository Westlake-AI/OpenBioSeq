import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (build_norm_layer,
                      constant_init, kaiming_init, normal_init)

from ..registry import NECKS


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m)
        elif isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            constant_init(m, val=1, bias=0)


@NECKS.register_module
class AvgPoolNeck(nn.Module):
    """Average pooling 1D or 2D average pooling neck."""

    def __init__(self, output_size=1):
        super(AvgPoolNeck, self).__init__()

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        if x.dim() == 3:
            x = F.adaptive_avg_pool1d(x[0], 1)
        elif x.dim() == 4:
            x = F.adaptive_avg_pool2d(x[0], 1)
        else:
            return x
        return [x]


@NECKS.register_module
class LinearNeck(nn.Module):
    """The linear neck: fc only.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_avg_pool=True):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.fc = nn.Linear(in_channels, out_channels)

    def init_weights(self, init_linear='normal', **kwargs):
        _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        return [self.fc(x)]


@NECKS.register_module
class ODCNeck(nn.Module):
    """The non-linear neck of ODC: fc-bn-relu-dropout-fc-relu.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 norm_cfg=dict(type='SyncBN'),
                 with_avg_pool=True):
        super(ODCNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.fc0 = nn.Linear(in_channels, hid_channels)
        self.bn0 = build_norm_layer(
            dict(**norm_cfg, momentum=0.001, affine=False), hid_channels)[1]
        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()

    def init_weights(self, init_linear='normal', **kwargs):
        _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class MoCoV2Neck(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global
            average pooling after backbone. Defaults to True.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(MoCoV2Neck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal', **kwargs):
        _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        return [self.mlp(x)]


@NECKS.register_module()
class NonLinearNeck(nn.Module):
    """The non-linear neck for SimCLR and BYOL.

    Structure: fc-bn-[relu-fc-bn] where the substructure in [] can be repeated.
    For the default setting, the repeated time is 1.
    The neck can be used in many algorithms, e.g., SimCLR, BYOL, SimSiam.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_layers (int): Number of fc layers. Defaults to 2.
        with_bias (bool): Whether to use bias in fc layers (except for the
            last). Defaults to False.
        with_last_bn (bool): Whether to add the last BN layer.
            Defaults to True.
        with_last_bn_affine (bool): Whether to have learnable affine parameters
            in the last BN layer (set False for SimSiam). Defaults to True.
        with_last_bias (bool): Whether to use bias in the last fc layer.
            Defaults to False.
        with_avg_pool (bool): Whether to apply the global average pooling
            after backbone. Defaults to True.
        vit_backbone (bool): Whether to use ViT (use cls_token) backbones. The
            cls_token will be removed in this neck. Defaults to False.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_bias=False,
                 with_last_bn=True,
                 with_last_bn_affine=True,
                 with_last_bias=False,
                 with_avg_pool=True,
                 vit_backbone=False,
                 norm_cfg=dict(type='SyncBN'),
                ):
        super(NonLinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.vit_backbone = vit_backbone
        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        self.bn0 = build_norm_layer(norm_cfg, hid_channels)[1]

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            if i != num_layers - 1:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(hid_channels, this_channels, bias=with_bias))
                self.add_module(f'bn{i}',
                                build_norm_layer(norm_cfg, this_channels)[1])
                self.bn_names.append(f'bn{i}')
            else:
                self.add_module(
                    f'fc{i}',
                    nn.Linear(
                        hid_channels, this_channels, bias=with_last_bias))
                if with_last_bn:
                    self.add_module(
                        f'bn{i}',
                        build_norm_layer(
                            dict(**norm_cfg, affine=with_last_bn_affine),
                            this_channels)[1])
                    self.bn_names.append(f'bn{i}')
                else:
                    self.bn_names.append(None)
            self.fc_names.append(f'fc{i}')

    def init_weights(self, init_linear='normal', **kwargs):
        _init_weights(self, init_linear, **kwargs)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.vit_backbone:  # remove cls_token
            x = x[-1]
        if self.with_avg_pool:
            if x.dim() == 3:
                x = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
            elif x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        else:
            x = x.reshape(x.size(0), -1)
        x = self.bn0(self.fc0(x))
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = fc(self.relu(x))
            if bn_name is not None:
                bn = getattr(self, bn_name)
                x = bn(x)
        return [x]


@NECKS.register_module()
class SwAVNeck(nn.Module):
    """The non-linear neck of SwAV: fc-bn-relu-fc-normalization.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        with_avg_pool (bool): Whether to apply the global average pooling after
            backbone. Defaults to True.
        with_l2norm (bool): whether to normalize the output after projection.
            Defaults to True.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='SyncBN').
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 with_l2norm=True,
                 norm_cfg=dict(type='SyncBN'),
                ):
        super(SwAVNeck, self).__init__()
        self.with_l2norm = with_l2norm
        self.with_avg_pool = with_avg_pool
        if out_channels == 0:
            self.projection_neck = None
        elif hid_channels == 0:
            self.projection_neck = nn.Linear(in_channels, out_channels)
        else:
            self.bn = build_norm_layer(norm_cfg, hid_channels)[1]
            self.projection_neck = nn.Sequential(
                nn.Linear(in_channels, hid_channels), self.bn,
                nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal', **kwargs):
        _init_weights(self, init_linear, **kwargs)

    def forward_projection(self, x):
        if self.projection_neck is not None:
            x = self.projection_neck(x)
        if self.with_l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
        return x

    def forward(self, x):
        # forward computing
        # x: list of feature maps, len(x) according to len(num_crops)
        avg_out = []
        for _x in x:
            _x = _x[0]
            if self.with_avg_pool:
                if _x.dim() == 3:
                    _x = F.adaptive_avg_pool1d(_x, 1).view(_x.size(0), -1)
                elif _x.dim() == 4:
                    _x = F.adaptive_avg_pool2d(_x, 1).view(_x.size(0), -1)
            else:
                _x = _x.reshape(_x.size(0), -1)
            avg_out.append(_x)
        feat_vec = torch.cat(avg_out)  # [sum(num_crops) * N, C]
        feat_vec = feat_vec.view(feat_vec.size(0), -1)
        output = self.forward_projection(feat_vec)
        return [output]


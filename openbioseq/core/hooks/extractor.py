import torch.nn as nn
from torch.utils.data import Dataset

from openbioseq import datasets
from openbioseq.utils import nondist_forward_collect, dist_forward_collect


class Extractor(object):
    """Feature extractor.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        samples_per_gpu (int): Number of images on each GPU, i.e., batch size
            of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        dist_mode (bool): Use distributed extraction or not. Default: False.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu,
                 workers_per_gpu,
                 forward_mode='extract',
                 dist_mode=False,
                 **kwargs):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                f'dataset must be a Dataset object or a dict, not {type(dataset)}')
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            samples_per_gpu,
            workers_per_gpu,
            dist=dist_mode,
            shuffle=False,
            prefetch=kwargs.get('prefetch', False),
            sample_norm_cfg=kwargs.get('sample_norm_cfg', dict()),
        )
        assert forward_mode in ['test', 'vis', 'extract',]
        self.forward_mode = forward_mode
        self.dist_mode = dist_mode
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_func(self, runner, **x):
        backbone_feat = runner.model(mode=self.forward_mode, **x)
        last_layer_feat = runner.model.module.neck([backbone_feat[-1]])[0]
        last_layer_feat = last_layer_feat.view(last_layer_feat.size(0), -1)
        return dict(feature=last_layer_feat.cpu())

    def __call__(self, runner):
        func = lambda **x: self._forward_func(runner, **x)
        if self.dist_mode:
            feats = dist_forward_collect(
                func,
                self.data_loader,
                runner.rank,
                len(self.dataset),
                ret_rank=-1)['feature']  # NxD
        else:
            feats = nondist_forward_collect(func, self.data_loader,
                                            len(self.dataset))['feature']
        return feats

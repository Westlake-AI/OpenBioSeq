from openbioseq.utils import build_from_cfg
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .registry import DATASETS, DATASOURCES


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_datasource(cfg):
    return build_from_cfg(cfg, DATASOURCES)

import torch

from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class ExtractDataset(BaseDataset):
    """The dataset outputs one view of an image or feature extraction.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(ExtractDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        data = self.data_source.get_sample(idx)
        data = self.pipeline(data)
        if self.prefetch:
            data = torch.from_numpy(to_numpy(data))
        return dict(data=data, idx=idx)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplementedError

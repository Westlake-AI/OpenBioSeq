import torch

from PIL import Image
from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class DeepClusterDataset(BaseDataset):
    """Dataset for DC and ODC.

    The dataset initializes clustering labels and assigns it during training.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[dict]): A list of dict, where each element represents
            an operation defined in `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(DeepClusterDataset, self).__init__(data_source, pipeline, prefetch)
        # init clustering labels
        self.labels = [-1 for _ in range(self.data_source.get_length())]

    def __getitem__(self, idx):
        data = self.data_source.get_sample(idx)
        assert isinstance(data, Image.Image), \
            'The output from the data source must be an Image, got: {}. \
            Please ensure that the list file does not contain labels.'.format(
            type(data))
        label = self.labels[idx]
        data = self.pipeline(data)
        if self.prefetch:
            data = torch.from_numpy(to_numpy(data))
        return dict(data=data, pseudo_label=label, idx=idx)

    def assign_labels(self, labels):
        assert len(self.labels) == len(labels), \
            "Inconsistent lenght of asigned labels, \
            {} vs {}".format(len(self.labels), len(labels))
        self.labels = labels[:]

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplementedError

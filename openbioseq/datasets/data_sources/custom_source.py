from abc import ABCMeta, abstractmethod

import os
from mmcv.utils import scandir
import torch

from ..registry import DATASOURCES


class Custom_base(metaclass=ABCMeta):

    EXTENSIONS = ('.pth', '.tar', '.pt',)
    CLASSES = None

    def __init__(self,
                 root, split, return_label=True, data_type="classification"):
        self.root = root
        self.split = split
        self.return_label = return_label
        self.data_type = data_type
        assert data_type in ["classification", "regression",]
        self.data = dict()
        self.targets = dict()
        self.labels = None
        self.load_data()

    @abstractmethod
    def load_data(self):
        pass

    def get_length(self):
        return len(self.labels)

    def get_sample(self, idx, split_override=None):
        split = self.split if split_override is None else split_override
        data = self.data[split][idx]
        if self.return_label:
            target = self.targets[split][idx]
            return data, target
        else:
            return data


@DATASOURCES.register_module
class ProcessedDataset(Custom_base):

    CLASSES = None

    def __init__(self,
                 root, split, return_label=True, data_type="classification"):
        super().__init__(root, split, return_label, data_type)

    @staticmethod
    def find_data_keys(keys):
        data_k = list(set(['data', 'sample', 'samples', 'x']) & set(keys))[0]
        label_k = list(set(['target', 'targets', 'label', 'labels', 'y']) & set(keys))[0]
        return data_k, label_k

    def load_data(self):
        for name in scandir(self.root, self.EXTENSIONS, False):
            data = torch.load(os.path.join(self.root, name))
            assert isinstance(data, dict)
            data_k, label_k = self.find_data_keys(data.keys())
            name = name.split(".")[0]
            self.data[name] = data[data_k].type(torch.float32)
            if self.data_type == "classification":
                self.targets[name] = data[label_k].type(torch.LongTensor)
            else:
                self.targets[name] = data[label_k].type(torch.float32)
        
        assert self.split in self.data.keys(), f"Invalid split {self.split}"
        self.labels = self.targets[self.split]

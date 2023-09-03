from collections import Counter

import os
import torch

from torch.nn import ConstantPad1d
from itertools import product
from tqdm import tqdm
from openbioseq.utils import print_log
from ..registry import DATASOURCES
from ..utils import read_file

try:
    from genomic_benchmarks import data_check
    from genomic_benchmarks.loc2seq import download_dataset
    from genomic_benchmarks.dataset_getters.pytorch_datasets import get_dataset
    from genomic_benchmarks.dataset_getters.utils import LetterTokenizer
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import vocab
except ImportError:
    data_check, download_dataset, get_dataset, LetterTokenizer = None, None, None, None
    get_tokenizer, vocab = None, None


@DATASOURCES.register_module
class GenomicDataset(object):
    """The implementation for loading Genomic Benchmark datasets.

    Args:
        root (str): Root path to string files.
        data_name (str): Name of dataset in Genomic Benchmarks.
        return_label (bool): Whether to return supervised labels.
        data_type (str): Type of the data.
    """

    CLASSES = None
    VOCAB = ['A', 'C', 'G', 'T']

    def __init__(self,
                 root="data",
                 data_name="",
                 split="train",
                 has_labels=True,
                 return_label=True,
                 data_type="classification",
                 max_seq_length=512,
                 max_data_length=None):
        # assert os.path.exists(root)
        assert data_type in ["classification", "regression",]
        assert split in ["train", "test",]
        self.data_name = data_name
        self.base_path = os.path.join(root, data_name, split)
        if not os.path.exists(root):
            download_dataset(data_name, local_repo=True, dest_path=root)

        self.tokenizer = get_tokenizer(LetterTokenizer())
        counter = Counter()
        for i in range(len(self.VOCAB)):
            counter.update(self.tokenizer(self.VOCAB[i]))
        self.vocabulary = vocab(counter)
        self.vocabulary.append_token("<pad>")
        self.vocabulary.insert_token("<unk>", 0)
        self.vocabulary.set_default_index(0)
        print("vocab len:" , self.vocabulary.__len__())
        print(self.vocabulary.get_stoi())

        self.data_list = []
        self.data = []
        self.labels = []
        label_mapper = {}

        for i, x in enumerate(os.listdir(self.base_path)):
            label_mapper[x] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.data_list.append(x)
                with open(x, "r") as f:
                    self.data.append(f.read())
                self.labels.append(label_mapper[label_type])

        print_log("Total file length: {}".format(self.data_list), logger='root')

    def get_length(self):
        return len(self.data_list)

    def get_sample(self, idx):
        seq = self.data[idx]
        target = self.labels[idx]

        # tokenize
        seq = [vocab[token] for token in tokenizer(seq)]
        # padding or truncate
        seq_len = len(x)
        if seq_len < self.max_seq_length:
            PAD_IDX = vocab["<pad>"]
            pad = ConstantPad1d((0, self.max_seq_length - seq_len), PAD_IDX)
            x = pad(x)
        else:
            x = x[:self.max_seq_length]

        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return seq, target
        else:
            return seq

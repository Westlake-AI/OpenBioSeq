import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..registry import DATASOURCES
from ..utils import read_file


def binarize(data_list, mapping):
    assert isinstance(data_list, list) and len(data_list) > 0
    token_list = list()
    num_entries = max(mapping.values()) + 1
    for _seq in data_list:
        try:
            onehot_seq = torch.zeros(
                num_entries, (len(_seq)), dtype=torch.float32)
            for _idx, _str in enumerate(_seq):
                map_idx = mapping[_str]
                onehot_seq[map_idx, _idx] = 1
            token_list.append(onehot_seq)
        except:
            print(f"Error seq:", _seq)
    return token_list


class TokenizeDataset(Dataset):
    """ Tokenize string to binary encoding """

    def __init__(self, data, mapping):
        super().__init__()
        self.data = data
        self.mapping = mapping
        self.num_entries = max(mapping.values()) + 1
           
    def __getitem__(self, idx):
        _seq = self.data[idx]
        onehot_seq = torch.zeros(self.num_entries, (len(_seq)), dtype=torch.float32)
        try:
            for _idx, _str in enumerate(_seq):
                map_idx = self.mapping[_str]
                onehot_seq[map_idx, _idx] = 1
        except:
            print(f"Error seq:", _seq)
        return onehot_seq

    def __len__(self):
        return len(self.data)


@DATASOURCES.register_module
class BioSeqDataset(object):
    """The implementation for loading any bio seqences.

    Args:
        root (str): Root path to string files.
        file_list (list or None): List of file names for N-fold cross
            validation training, e.g., file_list=['train_1.txt',].
        word_splitor (str): Split the data string.
        data_splitor (str): Split each seqence in the data.
        mapping_name (str): Predefined mapping for the bio string.
        return_label (bool): Whether to return supervised labels.
        data_type (str): Type of the data.
    """

    CLASSES = None

    ACGT = dict(A=0, C=1, G=2, T=3)
    AminoAcids = dict()

    def __init__(self,
                 root,
                 file_list=None,
                 word_splitor="",
                 data_splitor=" ",
                 mapping_name="ACGT",
                 return_label=True, data_type="classification"):
        assert file_list is None or isinstance(file_list, list)
        assert word_splitor in ["", " ", ",", ";", ".",]
        assert data_splitor in [" ", ",", ";", ".", "\t",]
        assert word_splitor != data_splitor
        assert mapping_name in ["ACGT", "AminoAcids",]
        assert data_type in ["classification", "regression",]

        # load files
        assert os.path.exists(root)
        if file_list is None:
            file_list = os.listdir(root)
        lines = list()
        for file in file_list:
            lines += read_file(os.path.join(root, file))
        self.has_labels = len(lines[0].split(data_splitor)) >= 2
        self.return_label = return_label
        self.data_type = data_type

        # preprocess
        if self.has_labels:
            data, self.labels = zip(*[l.strip().split(data_splitor)[-2:] for l in lines])
            if self.data_type == "classification":
                self.labels = [int(l) for l in self.labels]
                self.labels = torch.tensor(self.labels).type(torch.LongTensor)
            else:
                self.labels = [float(l) for l in self.labels]
                self.labels = torch.tensor(self.labels).type(torch.float32)
        else:
            # assert self.return_label is False
            self.labels = None
            data = [l.strip()[-1:] for l in lines]

        mapping = getattr(self, mapping_name)
        max_file_len = 100000
        num_workers = max(6, int(max_file_len / max_file_len + 1))

        if num_workers <= 1:
            self.data = binarize(data, mapping)
        else:
            tokens = None
            tokenizer = TokenizeDataset(data, mapping)
            process_loader = DataLoader(tokenizer, batch_size=num_workers * 1000,
                                        shuffle=False, num_workers=num_workers)
            for i, _tokens in tqdm(enumerate(process_loader)):
                if i == 0:
                    tokens = _tokens
                else:
                    tokens = torch.cat([tokens, _tokens])
            self.data = tokens

    def get_length(self):
        return len(self.data)

    def get_sample(self, idx):
        seq = self.data[idx]
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return seq, target
        else:
            return seq

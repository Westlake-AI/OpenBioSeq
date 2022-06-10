import os
import torch

from ..registry import DATASOURCES


ACGT = dict(
    A=0,
    C=1,
    G=2,
    T=3,
)

AminoAcids = dict()


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

    def __init__(self,
                 root,
                 file_list=None,
                 word_splitor="",
                 data_splitor=" ",
                 mapping_name="ACGT",
                 return_label=True, data_type="classification"):
        assert file_list is None or isinstance(file_list, list)
        assert word_splitor in ["", " ", ",", ";", "."]
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
            with open(os.path.join(root, file), 'r') as fp:
                lines += fp.readlines()
            fp.close()
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
        
        mapping = eval(mapping_name)
        num_entries = max(mapping.values()) + 1
        self.data = list()
        for _seq in data:
            onehot_seq = torch.zeros(num_entries, (len(_seq)), dtype=torch.float32)
            for _idx, _str in enumerate(_seq):
                map_idx = mapping[_str]
                onehot_seq[map_idx, _idx] = 1
            self.data.append(onehot_seq)

    def get_length(self):
        return len(self.data)

    def get_sample(self, idx):
        seq = self.data[idx]
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return seq, target
        else:
            return seq

import os
import torch

from tqdm import tqdm
from openbioseq.utils import print_log
from ..registry import DATASOURCES
from ..utils import read_file


@DATASOURCES.register_module
class DNASeqDataset(object):
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

    ACGT = dict(N=0, A=1, C=2, G=3, T=4)
    col_names = ['pos1', 
                 'pos2', 
                 'pos3', 
                 'g_umi_count', 
                 'r_umi_count', 
                 'g_total_count', 
                 'r_total_count', 
                 '1', 
                 '2', 
                 '3', 
                 '4', 
                 'seq', 
                 'umi', 
                 'total']
    AminoAcids = dict()

    def __init__(self,
                 root,
                 file_list=None,
                 word_splitor="",
                 data_splitor=" ",
                 mapping_name="ACGT",
                 has_labels=True,
                 target_type='',
                 filter_condition=0,
                 data_type="classification",
                 max_seq_length=1024,
                 max_data_length=None):
        assert file_list is None or isinstance(file_list, list)
        assert word_splitor in ["", " ", ",", ";", ".",]
        assert data_splitor in [" ", ",", ";", ".", "\t",]
        assert word_splitor != data_splitor
        assert mapping_name in ["ACGT", "AminoAcids",]
        assert data_type in ["classification", "regression",]
        assert target_type in ['umi', 'total']

        # load all files
        assert os.path.exists(root)
        if file_list is None:
            file_list = os.listdir(root)
        lines = list()
        for file in file_list:
            lines += read_file(os.path.join(root, file))

        # instance vars
        self.has_labels = len(lines[0].split(data_splitor)) >= 2 and has_labels
        self.data_type = data_type
        self.max_seq_length = max_seq_length
        self.filter_condition = filter_condition
        self.target_type = target_type

        print_log("Total file length: {}".format(len(lines)), logger='root')

        # preprocesing
        mapping = getattr(self, mapping_name) # mapping str to ints
        self.data_list, self.labels = [], []
        for l in tqdm(lines, desc='Data preprocessing:'):
            l = l.strip().split(data_splitor)

            # filtering
            con_g = int(l[self.col_names.index('g_total_count')]) > self.filter_condition
            con_r = int(l[self.col_names.index('r_total_count')]) > self.filter_condition
            con = con_g & con_r

            if con:
                if self.has_labels:
                    # data = [mapping[tok] for tok in l[self.col_names.index('seq')]] + [0] * padding
                    data_list = list(map(mapping.get, l[self.col_names.index('seq')]))
                    padding = self.max_seq_length - len(data_list)
                    if padding < 0:
                        data = data_list[:self.max_seq_length]
                    else:
                        data = data_list + [0] * padding

                    label = l[self.col_names.index(self.target_type)]
                    
                    if self.data_type == "classification":
                        label = torch.tensor(float(label)).type(torch.LongTensor)
                    else:
                        label = torch.tensor(float(label)).type(torch.float32)

                    self.labels.append(label)
                else:
                    # assert self.return_label is False
                    label = None
                    data = l.strip()[self.col_names.index['seq']]

                self.data_list.append(data)
                
        if max_data_length is not None:
            assert isinstance(max_data_length, (int, float))
            self.data = self.data[:max_data_length]
        print_log("Used data length: {}".format(len(self.data_list)), logger='root')
        

    def get_length(self):
        return len(self.data_list)

    def get_sample(self, idx):
        seq = self.data_list[idx]
        if self.has_labels:
            target = self.labels[idx]
            return seq, target
        else:
            return seq

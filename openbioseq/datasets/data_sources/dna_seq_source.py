import os
import torch
from itertools import product
from tqdm import tqdm
from openbioseq.utils import print_log
from ..registry import DATASOURCES
from ..utils import read_file


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers
    
    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.
    
    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    return kmer


@DATASOURCES.register_module
class DNASeqDataset(object):
    """The implementation for loading any bio seqences.

    Args:
        root (str): Root path to string files.
        file_list (list or None): List of file names for N-fold cross
            validation training, e.g., file_list=['train_1.txt',].
        word_splitor (str): Split the data string.
        data_splitor (str): Split each seqence in the data.
        return_label (bool): Whether to return supervised labels.
        data_type (str): Type of the data.
    """

    CLASSES = None
    toks = ['A', 'C', 'G', 'T']
    col_names = {'dna':['seq', 'label'],
                 'grna':['desc', 'lib', 'seq', 'label']}
    # col_names = ['pos1',
    #              'pos2',
    #              'pos3',
    #              'g_umi_count',
    #              'r_umi_count',
    #              'g_total_count',
    #              'r_total_count',
    #              '1',
    #              '2',
    #              '3',
    #              '4',
    #              'seq',
    #              'umi',
    #              'total']

    def __init__(self,
                 root,
                 file_list=None,
                 word_splitor="",
                 data_splitor=" ",
                 has_labels=True,
                 return_label=True,
                 k=6,
                 padding_idx=0,
                 data_type="classification",
                 seq_type='dna',
                 max_seq_length=512,
                 max_data_length=None):
        assert file_list is None or isinstance(file_list, list)
        assert word_splitor in ["", " ", ",", ";", ".",]
        assert data_splitor in [" ", ",", ";", ".", "\t",]
        assert word_splitor != data_splitor
        assert data_type in ["classification", "regression",]
        # assert target_type in ['umi', 'total']

        # load all files
        assert os.path.exists(root)
        if file_list is None:
            file_list = os.listdir(root)
        lines = list()
        for file in file_list:
            lines += read_file(os.path.join(root, file))

        # instance vars
        self.has_labels = len(lines[0].split(data_splitor)) >= 2 and has_labels
        self.return_label = return_label
        self.data_type = data_type
        self.max_seq_length = max_seq_length
        self.padding_idx = padding_idx
        self.kmer2idx = {''.join(x) : i for i, x in enumerate(product(self.toks, repeat=k), start=1)}
        print_log("Total file length: {}".format(len(lines)), logger='root')

        col_name = self.col_names[seq_type]
        # preprocesing
        self.data_list, self.labels = [], []
        for l in tqdm(lines, desc='Data preprocessing:'):
            l = l.strip().split(data_splitor)
            seq = l[col_name.index('seq')]
            kmer_seq = seq2kmer(seq, k=k)
            kmer_idx_seq = list(map(self.kmer2idx.get, kmer_seq))
            padding = self.max_seq_length - len(kmer_idx_seq)

            if padding < 0:
                data = kmer_idx_seq[:self.max_seq_length]
            else:
                data = kmer_idx_seq + [padding_idx] * padding

            if self.has_labels:
                label = l[col_name.index('label')]
                
                if self.data_type == "classification":
                    label = torch.tensor(float(label)).type(torch.LongTensor)
                else:
                    label = torch.tensor(float(label)).type(torch.float32)

                self.labels.append(label)
            else:
                # assert self.return_label is False
                label = None
                data = l.strip()[col_name.index['seq']]

            self.data_list.append(data)
                
        if max_data_length is not None:
            assert isinstance(max_data_length, (int, float))
            self.data = self.data[:max_data_length]
        print_log("Used data length: {}".format(len(self.data_list)), logger='root')
        

    def get_length(self):
        return len(self.data_list)

    def get_sample(self, idx):
        seq = self.data_list[idx]
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return seq, target
        else:
            return seq
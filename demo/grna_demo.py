"""
An example of prediction of gRNA editing efficiency.

Example command (a sequence in 63 bits):
python grna_demo.py TTGCTGTATCTCTTGCCAGGCCCAAGGCTGCAGAGGGAATTGGTAATATACTTCATTTAATAA

Output results:
0.20432067
"""

import argparse
import torch
from mmcv.runner import load_checkpoint

from openbioseq.datasets.data_sources.bio_seq_source import binarize
from openbioseq.models import build_model
from openbioseq.datasets.utils import read_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process an input gRNA sequence to predict')
    parser.add_argument('--input_seq', type=str, default=None, help='input sequence')
    parser.add_argument('--input_file', type=str, default=None,
                        help='path to a input file containing several sequences')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode of demo')
    args = parser.parse_args()
    return args


def get_model_config(seq_len=63, embed_dim=64, patch_size=2):
    """ Transformer """

    checkpoint = "https://github.com/Westlake-AI/OpenBioSeq/releases/download/v0.2.0/k562_layer4_p2_h4_d64_init_bs256_ep100.pth"
    model = dict(
        type='Classification',
        pretrained=None,
        backbone=dict(
            type='SequenceTransformer',
            arch=dict(
                embed_dims=embed_dim,
                num_layers=4,
                num_heads=4,
                feedforward_channels=embed_dim * 4),
            in_channels=4,
            patch_size=patch_size,
            seq_len=int(seq_len / patch_size) + bool(seq_len % patch_size != 0),
            norm_cfg=dict(type='LN', eps=1e-6),
            drop_rate=0.1,
            drop_path_rate=0.1,
            init_values=0.1,
            final_norm=True,
            out_indices=-1,  # last layer
            with_cls_token=False,
            output_cls_token=False),
        head=dict(
            type='RegHead',
            loss=dict(type='RegressionLoss', mode='huber_loss',
                loss_weight=1.0, reduction='mean',
                activate='sigmoid', alpha=0.2, gamma=1.0, beta=1.0, residual=False),
            with_avg_pool=True, in_channels=embed_dim, out_channels=1),
    )

    return model, checkpoint


def main():
    args = parse_args()
    if args.debug:
        input_seq = ["TTGCTGTATCTCTTGCCAGGCCCAAGGCTGCAGAGGGAATTGGTAATATACTTCATTTAATAA"]
    else:
        if args.input_seq is not None:
            input_seq = [args.input_seq]
        elif args.input_file is not None:
            input_seq = read_file(args.input_file)
            for i in range(len(input_seq)):
                input_seq[i] = input_seq[i].replace('\n', '')
        else:
            print(args)
            assert False and "Invalid input args"

    # input
    seq_len, key_num = 63, 4
    key_mapping = dict(A=0, C=1, G=2, T=3)
    try:
        input_seq = binarize(
            input_seq, mapping=key_mapping, max_seq_length=seq_len, data_splitor=',')
    except ValueError:
        assert False and "Please check the input sequence"

    # build the model and load checkpoint
    cfg_model, checkpoint = get_model_config(seq_len=seq_len)
    model = build_model(cfg_model)
    load_checkpoint(model, checkpoint, map_location='cpu')

    # inference
    if len(input_seq) == 1:
        input_seq = input_seq[0].unsqueeze(0)
    else:
        input_seq = torch.concat(input_seq).view(-1, key_num, seq_len)

    output = model(input_seq, mode='inference').detach().cpu().numpy()
    print("Prediction:", output)


if __name__ == '__main__':
    main()

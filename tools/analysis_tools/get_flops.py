"""
An example to count Params and FLOPs.

Example command:
python tools/get_flops.py [PATH_TO_config] --channel 4 --shape 512
"""
import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from openbioseq.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--channel',
        type=int,
        default=4,
        help='input data channel')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512],
        help='input data size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    in_channel = args.channel
    if len(args.shape) == 1:
        input_shape = (in_channel, args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (in_channel, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    if args.channel == 0:  # using nn.Embedding in the model
        input_shape = input_shape[1:]

    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model)
    model.eval()

    if hasattr(model, 'forward_backbone'):
        model.forward = model.forward_backbone
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()

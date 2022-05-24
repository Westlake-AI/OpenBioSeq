import json
import os
import argparse

from datetime import datetime
from mmcv import Config
# from numpy.core.fromnumeric import prod, var
# from tools.train import main
# from openselfsup.apis import train
from functools import reduce
from operator import getitem
from itertools import product


class ConfigGenerator:
    def __init__(self, base_path: str, num_device: int) -> None:
        self.base_path = base_path
        self.num_device = num_device

    def _path_parser(self, path: str) -> str:
        assert isinstance(path, str)
        base_dir = os.path.join(*self.base_path.split('/')[:-1])
        base_name = self.base_path.split('/')[-1] # file name
        base_prefix = base_name.split('.')[0] # prefix
        backbone = base_prefix.split('_')[0]

        return base_dir, backbone, base_prefix

    def _combinations(self, var_dict: dict) -> list:
        assert isinstance(var_dict, dict)
        ls = list(var_dict.values())
        cbs = [x for x in product(*ls)] # all combiantions

        return cbs

    def set_nested_item(self, dataDict: dict, mapList: list, val) -> dict:
        """Set item in nested dictionary"""
        reduce(getitem, mapList[:-1], dataDict)[mapList[-1]] = val

        return dataDict

    def generate(self, model_var: dict, gm_var: dict, abbs: dict) -> None:
        assert isinstance(model_var, dict)
        assert isinstance(gm_var, dict)
        cfg = dict(Config.fromfile(self.base_path))
        base_dir, backbone, base_prefix = self._path_parser(self.base_path) # analysis path
        model_cbs = self._combinations(model_var)
        gm_cbs = self._combinations(gm_var)

        # params for global .sh file
        port = 99999
        time = datetime.today().strftime('%Y%m%d_%H%M%S')
        with open('{}_{}.sh'.format(os.path.join(base_dir, base_prefix), time), 'a') as shfile:
            # model setting
            for c in model_cbs:
                cfg_n = cfg # reset
                config_dir = os.path.join(base_dir, backbone)
                for i, kv in enumerate(zip(list(model_var.keys()), c)):
                    k = kv[0].split('.')
                    v = kv[1]
                    cfg_n = self.set_nested_item(cfg_n, k, v) # assign value
                    config_dir += '/{}{}'.format(str(k[-1]), str(v))
                comment = ' '.join(config_dir.split('/')[-i-1:]) # e.g. alpha1.0 mask_layer 1
                shfile.write('# {}\n'.format(comment))

                # base setting
                for b in gm_cbs:
                    base_params = ''
                    for kv in zip(list(gm_var.keys()), b):
                        a = kv[0].split('.')
                        n = kv[1]
                        cfg_n = self.set_nested_item(cfg_n, a, n)
                        base_params += '_{}{}'.format(str(a[-1]), str(n))

                    # saving json config
                    config_dir = config_dir.replace('.', '_')
                    base_params = base_params.replace('.', '_')
                    for word, abb in abbs.items():
                        base_params = base_params.replace(word, abb)
                    if not os.path.exists(config_dir):
                        os.makedirs(config_dir)
                    file_name = os.path.join(config_dir, '{}{}.json'.format(base_prefix, base_params))
                    with open(file_name, 'w') as configfile:
                        json.dump(cfg, configfile, indent=4)

                    # write cmds for .sh
                    port += 1
                    cmd = 'CUDA_VISIBLE_DEVICES=0 PORT={} bash tools/dist_train.sh {} {} &\nsleep 1s \n'.format(port, file_name, self.num_device)
                    shfile.write(cmd)
                shfile.write('\n')
    print('Generation completed.')


def main():
    base_path = 'configs/classification/cifar100/automix/auto_train/resnext50_automix.py'

    # emix
    abbs = {
            'total_epochs': 'ep'
           }
    # create nested dirs
    model_var = {
                # 'model.grad_mode': [None, 'plain', 'concat', 'softmax', 'minmax'],
                # 'model.mask_loss': [10, 25],
                # 'model.mix_block.mode': ['embedded_gaussian'],
                'model.momentum': [0.999, 0.996],
                # 'model.mix_block.lam_concat_theta': [True, False],
                # 'model.mix_block.lam_concat_g': [True],
                # 'model.mix_block.loss_mask': ['L1', "L2"],
                'model.mix_block.loss_mask': ["L2",],
                # 'model.mix_block.loss_mask': ["none",],
                # 'model.mix_block.loss_mask': ["Variance"],
                # 'model.mix_block.lam_mul': [True,],
                # 'model.mix_block.all_concat_phi': [True],
                # 'model.mix_block.all_concat': [True],
                # 'model.mix_block.mask_multi': [True, False],
            }
    gm_var = {
            #   'optimizer.paramwise_options.mix_block.lr_mult': [5, 10, 20, 25],
            #   'optimizer.paramwise_options.mix_block.weight_decay': [0.0001, 0.0005],
              'lr_config.min_lr': [1e-1],
            # 'lr_config.min_lr': [5e-4, 1e-4, 0.],  # CUB
            'total_epochs': [800, 1200]
        }

    # baselines
    # abbs = {'total_epochs': 'ep'}

    # model_var = {'model.alpha': [0.2],
    #              'model.mix_mode': ["CutMix", "SaliencyMix", "FMix"]}
    # gm_var = {'lr_config.min_lr' : [0],
    #           'total_epochs': [200, 400, 800, 1200]}
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()
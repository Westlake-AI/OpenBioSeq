import copy
import os
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script extracts backbone weights from a checkpoint')
    parser.add_argument('path', help='checkpoint file')
    args = parser.parse_args()
    return args


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    args = parse_args()
    path = args.path
    save_path = os.path.join("work_dirs/my_pretrains", path.split("work_dirs/")[1])
    mkdir(save_path)
    save_name = os.path.join(save_path, path.split('/')[-1])
    
    ck = torch.load(path, map_location=torch.device('cpu'))
    
    output_dict = dict(state_dict=dict(), author="openmixup")
    
    ck = ck['state_dict'] if ck.get('state_dict', None) is not None else ck
    ck = ck['model'] if ck.get('model', None) is not None else ck

    for key, value in ck.items():
        new_key = copy.copy(key)
        # remove backbone keys
        for prefix_k in ['encoder', 'backbone', 'timm_model',]:
            if new_key.startswith(prefix_k):
                new_key = new_key[len(prefix_k) + 1: ]

        # replace timm to openmixup
        if new_key.find('patch_embed.proj.') != -1:
            new_key = new_key.replace('patch_embed.proj.', 'patch_embed.projection.')
        if new_key.find('mlp.fc1.') != -1:
            new_key = new_key.replace('mlp.fc1.', 'ffn.layers.0.0.')
        if new_key.find('mlp.fc2.') != -1:
            new_key = new_key.replace('mlp.fc2.', 'ffn.layers.1.')
        
        if new_key.find('blocks') != -1:
            new_key = new_key.replace('blocks', 'layers')
        if new_key.find('.norm') != -1:
            new_key = new_key.replace('.norm', '.ln')
        if new_key == 'norm.weight':
            new_key = 'ln1.weight'
        if new_key == 'norm.bias':
            new_key = 'ln1.bias'
        
        output_dict['state_dict'][new_key] = value
        print("keep key {} -> {}".format(key, new_key))

    torch.save(output_dict, save_name)
    print("save ckpt:", save_name)


if __name__ == '__main__':
    main()

# usage exam:
# python tools/extract_dir_weights.py /public/home/liziqinggroup/liuzicheng/src/FineGrained_v0319/work_dirs/selfsup/byol/cub/

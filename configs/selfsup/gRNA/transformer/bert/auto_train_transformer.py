from openbioseq.utils import ConfigGenerator


def main():

    # BERT
    # base_path = "configs/selfsup/gRNA/transformer/bert/layer4_p2_h4_d64_init_bs256.py"
    base_path = "configs/selfsup/gRNA/transformer/bert/layer4_p2_h4_d64_init_focal_bs256.py"
    # base_path = "configs/selfsup/gRNA/transformer/bert/layer8_p2_h8_d128_init_bs256.py"

    # abbreviation of long attributes
    abbs = {
        'runner.max_epochs': 'ep',
    }
    # create nested dirs (cannot be none)
    model_var = {
        # 'model.backbone.mask_ratio': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,],
        # 'model.backbone.mask_ratio': [0.15, 0.2, 0.3,],
        # 'model.mask_ratio': [0.1, 0.15, 0.2, 0.25,],
        'model.mask_ratio': [0.15, 0.2, 0.25, 0.3,],
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        'model.backbone.mask_token': ['learnable', 'zero',],
        'optimizer.lr': [1e-3,],
        # 'optimizer.lr': [3e-3, 1e-3,],
        'lr_config.warmup_iters': [40,],
        # 'optimizer.weight_decay': [1e-2, 5e-2,],
        # 'lr_config.min_lr': [1e-5, 1e-6],
        'runner.max_epochs': [400,],
        # 'runner.max_epochs': [800,],
    }

    num_device = 1

    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()
from openseq.utils import ConfigGenerator


def main():
    """Automatic Config Generator: generate openmixup configs in terms of keys"""

    base_path = "configs/classification/processed/EMG/cnn/cnn5/cnn5_bs128.py"

    # abbreviation of long attributes
    abbs = {
        'runner.max_epochs': 'ep'
    }
    # create nested dirs (cannot be none)
    model_var = {
        'model.backbone.patch_size': [4, 8, 12, 16,],
        # 'model.backbone.patch_size': [25,],
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        'model.backbone.drop_rate': [0.1, 0.2, 0.3, 0.35,],
        # 'optimizer.lr': [1e-3, 3e-3, 1e-4, 3e-4,],
        # 'optimizer.weight_decay': [1e-2, 5e-2],
        'model.backbone.kernel_size': [7, 9,],
        # 'lr_config.min_lr': [1e-6],
        'runner.max_epochs': [40,],
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()
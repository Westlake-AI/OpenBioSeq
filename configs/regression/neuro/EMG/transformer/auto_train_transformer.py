from openbioseq.utils import ConfigGenerator


def main():
    """Automatic Config Generator: generate openbioseq configs in terms of keys"""

    base_path = ""
    base_path = ""

    # abbreviation of long attributes
    abbs = {
        'runner.max_epochs': 'ep'
    }
    # create nested dirs (cannot be none)
    model_var = {
        # Loss
        # 'model.head.loss.mode': ["l1_loss", "mse_loss", "huber_loss", "charbonnier_loss",
        #     "focal_l1_loss", "focal_mse_loss", "balanced_mse_loss",],
        # 'model.head.loss.mode': ["huber_loss", "focal_mse_loss",],
        # 'model.head.loss.mode': ["l1_loss",],
        # 'model.head.loss.mode': ["focal_l1_loss",],
        # 'model.head.loss.mode': ["balanced_mse_loss",],
        'model.head.loss.mode': ["l1_loss", "focal_l1_loss", "huber_loss", "balanced_mse_loss",],
    }
    # adjust sub-attributes (cannot be none)
    gm_var = {
        # Loss
        # 'model.head.loss.residual': [False,],
        # 'model.head.loss.alpha': [0.2, 0.5, 1,],
        # 'model.head.loss.activate': ['sigmoid', 'tanh',],
        # augments
        # 'model.backbone.drop_path_rate': [0.1, 0.2,],
        # 'model.backbone.drop_path_rate': [0.2,],
        # 'model.backbone.drop_path_rate': [0.1,],
        # Optimizer
        # 'optimizer.lr': [3e-3, 1e-3, 3e-4, 1e-4,],
        'optimizer.lr': [3e-3, 1e-3, 3e-4,],
        # 'optimizer.lr': [1e-3, 3e-3,],
        # 'optimizer.lr': [3e-3,],
        'optimizer.weight_decay': [1e-2, 5e-2],
        # 'optimizer.weight_decay': [5e-2,],
        # 'lr_config.min_lr': [1e-6, 1e-5,],
        'runner.max_epochs': [100,],
    }
    
    num_device = 1
    
    generator = ConfigGenerator(base_path=base_path, num_device=num_device)
    generator.generate(model_var, gm_var, abbs)


if __name__ == '__main__':
    main()
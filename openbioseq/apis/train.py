import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_runner, DistSamplerSeedHook

from openbioseq.datasets import build_dataloader
from openbioseq.core.hooks import (build_hook, build_addtional_scheduler, build_optimizer,
                             DistOptimizerHook, Fp16OptimizerHook)
from openbioseq.utils import get_root_logger, print_log

# import fp16 supports
try:
    import apex
    default_fp16 = 'apex'
except ImportError:
    default_fp16 = 'mmcv'
    warnings.warn('DeprecationWarning: Nvidia Apex is not installed, '
                  'using FP16OptimizerHook modified from mmcv.')


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                timestamp=None,
                meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=cfg.gpus,
            dist=distributed,
            sampler=cfg.sampler,
            shuffle=True,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            prefetch=cfg.prefetch,
            persistent_workers=getattr(cfg, 'persistent_workers', True),
            sample_norm_cfg=cfg.sample_norm_cfg) for ds in dataset
    ]

    # if you have addtional_scheduler, select chosen params
    if cfg.get('addtional_scheduler', None) is not None:
        param_names = dict(model.named_parameters()).keys()
        assert isinstance(cfg.optimizer.get('paramwise_options', False), dict)
    
    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # fp16 and optimizer
    if distributed:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
        if cfg.get('use_fp16', False):
            # fp16 settings
            fp16_cfg = cfg.get('fp16', dict(type=None))
            fp16_cfg['type'] = fp16_cfg.get('type', default_fp16)
            fp16_cfg['loss_scale'] = fp16_cfg.get(
                'loss_scale', dict(init_scale=512., mode='dynamic'))
            if fp16_cfg['type'] == 'apex':
                model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level="O1")
                print_log('**** Initializing mixed precision apex done. ****')
            elif fp16_cfg['type'] == 'mmcv':
                optimizer_config = Fp16OptimizerHook(
                    **cfg.optimizer_config, **fp16_cfg, distributed=True)
                print_log('**** Initializing mixed precision mmcv done. ****')
    else:
        optimizer_config = cfg.optimizer_config
    
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model if next(model.parameters()).is_cuda else model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))
    
    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # preprocess hooks: add EMAHook bofore ValidationHook and CheckpointSaverHook
    for hook in cfg.get('custom_hooks', list()):
        if hook.type == "EMAHook":
            common_params = dict(dist_mode=True)
            runner.register_hook(build_hook(hook, common_params), priority='NORMAL')
    
    # register basic hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    if distributed:
        runner.register_hook(DistSamplerSeedHook())
    
    # register custom hooks
    for hook in cfg.get('custom_hooks', list()):
        common_params = dict(dist_mode=distributed)
        if hook.type == "DeepClusterAutoMixHook" or hook.type == "DeepClusterHook":
            common_params = dict(dist_mode=distributed, data_loaders=data_loaders)
        elif hook.type == "EMAHook":
            continue
        runner.register_hook(build_hook(hook, common_params), priority='NORMAL')
    # register custom optional_scheduler hook
    if cfg.get('addtional_scheduler', None) is not None:
        runner.register_hook(
            build_addtional_scheduler(param_names, cfg.addtional_scheduler))

    # register evaluation hook
    if cfg.get('evaluation', None):
        eval_cfg = cfg.get('evaluation', dict())
        eval_cfg = dict(
            type='ValidateHook',
            dataset=cfg.data.val,
            dist_mode=distributed,
            initial=eval_cfg.get('initial', True),
            interval=eval_cfg.get('interval', 1),
            save_val=eval_cfg.get('save_val', False),
            samples_per_gpu=eval_cfg.get('samples_per_gpu', cfg.data.samples_per_gpu),
            workers_per_gpu=eval_cfg.get('samples_per_gpu', cfg.data.workers_per_gpu),
            eval_param=eval_cfg.get('eval_param', dict(topk=(1, 5))),
            prefetch=cfg.data.val.get('prefetch', False),
            sample_norm_cfg=cfg.sample_norm_cfg,
        )
        # We use `ValidationHook` instead of `EvalHook` in mmcv. `EvalHook` needs to be
        # executed after `IterTimerHook`, or it will cause a bug if use `IterBasedRunner`.
        runner.register_hook(build_hook(eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    cfg.workflow = [tuple(x) for x in cfg.workflow]
    runner.run(data_loaders, cfg.workflow)

_base_ = [
    '../../../../_base_/datasets/neuro/Epilepsy_metric.py',
    '../../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='BarlowTwins',
    backbone=dict(
        type='PlainCNN',
        depth=5,
        in_channels=1,
        patch_size=8,
        kernel_size=7,
        base_channels=64, out_channels=512,
        drop_rate=0.1,
        out_indices=(3,),  # no conv-1, x-1: stage-x
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=512, hid_channels=2048, out_channels=2048,
        num_layers=3,
        with_bias=True, with_last_bn=False, with_last_bn_affine=False,
        with_avg_pool=True),
    head=dict(
        type='LatentCrossCorrelationHead',
        in_channels=2048)
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-3,
    weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lr_mult=0.025),
        'bias': dict(weight_decay=0., lr_mult=0.025),
    })
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# checkpoint
checkpoint_config = dict(interval=100, max_keep_ckpts=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

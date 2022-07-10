_base_ = [
    '../../../../_base_/datasets/neuro/sleepEDF_swav_no_queue_metric.py',
    '../../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SwAV',
    backbone=dict(
        type='PlainCNN',
        depth=5,
        in_channels=1,
        patch_size=25,
        kernel_size=7,
        base_channels=64, out_channels=512,
        drop_rate=0.,
        out_indices=(3,),  # no conv-1, x-1: stage-x
    ),
    neck=dict(
        type='SwAVNeck',
        in_channels=512, hid_channels=1024, out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='SwAVHead',
        feat_dim=128,  # equal to neck['out_channels']
        epsilon=0.05,
        temperature=0.2,
        num_crops=[2, 2],
        num_prototypes=1000)
)

# SSL data
data = dict(
    samples_per_gpu=256, workers_per_gpu=4,
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-3,
    weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
optimizer_config = dict(
    cancel_grad=dict(prototypes=108),  # cancel grad of `prototypes` for 1 epoch
)

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

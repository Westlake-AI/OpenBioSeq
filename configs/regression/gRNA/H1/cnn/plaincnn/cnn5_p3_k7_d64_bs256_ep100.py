_base_ = [
    '../../../../_base_/datasets/gRNA/on_target_H1.py',
    '../../../../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='PlainCNN',
        depth=5,
        in_channels=4,
        patch_size=3,
        patchfied=False,
        kernel_size=7,
        base_channels=64, out_channels=512,
        drop_rate=0.1,
        out_indices=(3,),  # no conv-1, x-1: stage-x
    ),
    head=dict(
        type='RegHead',
        loss=dict(type='RegressionLoss',
            mode='mse_loss',
            loss_weight=1.0, reduction='mean',
        ),
        with_avg_pool=True, in_channels=512, out_channels=1),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6)

# checkpoint
checkpoint_config = dict(interval=100, max_keep_ckpts=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

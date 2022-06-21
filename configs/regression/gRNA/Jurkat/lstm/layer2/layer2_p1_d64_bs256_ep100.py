_base_ = [
    '../../../../_base_/datasets/gRNA/on_target_Jurkat.py',
    '../../../../_base_/default_runtime.py',
]

embed_dim = 64
num_layers = 2
patch_size = 1
seq_len = 63

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='SequenceLSTM',
        in_channels=4,
        patch_size=patch_size,
        seq_len=int(seq_len / patch_size) + bool(seq_len % patch_size != 0),
        embed_dims=embed_dim,
        num_layers=num_layers,
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_rate=0.2,
        final_norm=False,
        bidirectional=True,
        out_indices=-1,  # last layer
    ),
    head=dict(
        type='RegHead',
        loss=dict(type='RegressionLoss',
            mode='mse_loss',
            loss_weight=1.0, reduction='mean',
        ),
        with_avg_pool=True, in_channels=embed_dim * 4 * num_layers, out_channels=1),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-4,
    weight_decay=1e-2, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
optimizer_config = dict(
    update_interval=1, use_fp16=use_fp16)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6)

# checkpoint
checkpoint_config = dict(interval=100, max_keep_ckpts=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

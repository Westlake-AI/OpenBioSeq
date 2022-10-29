_base_ = [
    '../../../../_base_/datasets/gRNA/on_target_K562.py',
    '../../../../_base_/default_runtime.py',
]

embed_dim = 64
patch_size = 2
seq_len = 63

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='SequenceTransformer',
        arch=dict(
            embed_dims=embed_dim,
            num_layers=4,
            num_heads=4,
            feedforward_channels=embed_dim * 4,
        ),
        in_channels=4,
        patch_size=patch_size,
        seq_len=int(seq_len / patch_size) + bool(seq_len % patch_size != 0),
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_rate=0.1,
        drop_path_rate=0.1,
        init_values=0.1,
        final_norm=True,
        out_indices=-1,  # last layer
        with_cls_token=False,
        output_cls_token=False,
    ),
    head=dict(
        type='RegHead',
        loss=dict(type='RegressionLoss', mode='huber_loss',
            loss_weight=1.0, reduction='mean',
            activate='sigmoid', alpha=0.2, gamma=1.0, beta=1.0, residual=False),
        with_avg_pool=True, in_channels=embed_dim, out_channels=1),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-3,
    weight_decay=5e-2, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
        # 'noise_sigma': dict(weight_decay=0., lr_mult=1e-1),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(mode='dynamic'))
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=1)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# checkpoint
checkpoint_config = dict(interval=200, max_keep_ckpts=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

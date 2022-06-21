_base_ = [
    '../../../../_base_/datasets/neuro/cls.py',
    '../../../../_base_/default_runtime.py',
]

embed_dim = 64
patch_size = 2
seq_len = 50

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
        in_channels=16,
        patch_size=patch_size,
        seq_len=int(seq_len / patch_size) + bool(seq_len % patch_size != 0),
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_rate=0.1,
        drop_path_rate=0.,
        init_values=1e-2,
        final_norm=True,
        out_indices=-1,  # last layer
        frozen_stages=4,  # linear
        with_cls_token=False,
        output_cls_token=False,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, in_channels=embed_dim, num_classes=18)
)

# data
data_root = 'data/EMG_classification/'
data = dict(
    train=dict(
        data_source=dict(
            type='ProcessedDataset', root=data_root, split='train',
    )),
    val=dict(
        data_source=dict(
            type='ProcessedDataset', root=data_root, split='test',
    )),
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-4,
    weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0),
    update_interval=1, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-6,
    warmup='linear',
    warmup_iters=1, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)

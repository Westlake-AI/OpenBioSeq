_base_ = [
    '../../../_base_/datasets/gRNA/gRNA_pretrain.py',
    '../../../_base_/default_runtime.py',
]

embed_dim = 64
patch_size = 2
seq_len = 63

# model settings
model = dict(
    type='BERT',
    pretrained=None,
    backbone=dict(
        type='SimMIMTransformer',
        arch=dict(
            embed_dims=embed_dim,
            num_layers=4,
            num_heads=4,
            feedforward_channels=embed_dim * 4,
        ),
        in_channels=4,
        patch_size=patch_size,
        seq_len=int(seq_len / patch_size) + bool(seq_len % patch_size != 0),
        mask_layer=0,
        mask_ratio=0.15,  # BERT 15%
        mask_token='learnable',
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_rate=0.1,
        drop_path_rate=0.1,
        final_norm=True,
        out_indices=-1,  # last layer
        with_cls_token=True,
        output_cls_token=True,
    ),
    neck=dict(
        type='SimMIMNeck',
        feature_Nd="1d", in_channels=embed_dim, out_channels=4, encoder_stride=1),
    head=dict(
        type='MIMHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        feature_Nd="1d", unmask_weight=0., encoder_in_channels=4,
    )
)

# dataset
data = dict(samples_per_gpu=256, workers_per_gpu=4)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=1e-2, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale=dict(mode='dynamic'))
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

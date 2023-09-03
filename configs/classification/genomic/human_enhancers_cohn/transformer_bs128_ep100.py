_base_ = [
    '../../_base_/datasets/genomic/genomic_benchmark.py',
    '../../_base_/default_runtime.py',
]

embed_dim = 64
seq_len = 256

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='SequenceTransformer',
        arch={'embed_dims': embed_dim,
              'num_layers': 12,
              'num_heads': embed_dim // 16,
              'feedforward_channels': embed_dim * 4},
        in_channels=8,
        padding_index=0,
        seq_len=seq_len,
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_rate=0.1,
        drop_path_rate=0.1,
        init_values=0.1,
        final_norm=True,
        out_indices=-1,  # last layer
        with_cls_token=False,
        output_cls_token=False,
        with_embedding=True,
    ),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        with_avg_pool=True, multi_label=False, in_channels=512, num_classes=2)
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=3e-4,
    weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)

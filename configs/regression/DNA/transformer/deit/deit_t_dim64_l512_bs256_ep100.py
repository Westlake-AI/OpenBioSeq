_base_ = [
    '../../../_base_/datasets/DNA/dna.py',
    '../../../_base_/default_runtime.py',
]

embed_dim = 64
seq_len = 512

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
        in_channels=4096,
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
        type='RegHead',
        loss=dict(type='RegressionLoss', mode='huber_loss',
            loss_weight=1.0, reduction='mean',
            activate='sigmoid', alpha=0.2, gamma=1.0, beta=1.0, residual=False),
        with_avg_pool=True, in_channels=embed_dim, out_channels=1),
)

# dataset settings
data_root = 'data/dna/'
data_source_cfg = dict(
    type='DNASeqDataset',
    file_list=None,  # use all splits
    word_splitor=" ", data_splitor=",",  # gRNA tokenize
    data_type="regression", target_type='total',
    max_seq_length=512,
)
data = dict(
    samples_per_gpu=128,  # 256
    workers_per_gpu=4,
    train=dict(
        data_source=dict(root=data_root+"train", **data_source_cfg)),
    val=dict(
        data_source=dict(root=data_root+"test", **data_source_cfg)),
)
update_interval = 1

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-3,
    weight_decay=1e-2, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),
        'noise_sigma': dict(weight_decay=0., lr_mult=1e-1),
    })

# apex
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale='dynamic')
optimizer_config = dict(
    grad_clip=dict(max_norm=5.0), update_interval=update_interval)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=1, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)

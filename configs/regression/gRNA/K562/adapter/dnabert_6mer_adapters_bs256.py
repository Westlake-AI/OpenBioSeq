_base_ = [
    '../../../_base_/datasets/gRNA/on_target_K562_adaptor.py',
    '../../../_base_/default_runtime.py',
]

embed_dim = 768
seq_len = 63-5 # 6mer

# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='HuggingFaceBackbone',
        model_name='Bert',
        config_args='pretrained_model/DNABERT_6/config.json',
        pretrained='pretrained_model/DNABERT_6/',
        with_cls_token=False,
        output_cls_token=False,
        adapter_name='mam_adapter',
    ),
    head=dict(
        type='RegHead',
        loss=dict(type='RegressionLoss', mode='huber_loss',
            loss_weight=1.0, reduction='mean',
            activate='sigmoid', alpha=0.2, gamma=1.0, beta=1.0, residual=False),
        with_avg_pool=True, in_channels=embed_dim, out_channels=1),
)

# dataset settings
data_root = 'data/on_target_K562/'
data_source_cfg = dict(
    type='DNASeqDataset',
    file_list=None,  # use all splits
    word_splitor=" ", data_splitor="\t",  # gRNA tokenize
    data_type="regression", seq_type='grna',
    max_seq_length=seq_len,
)
data = dict(
    samples_per_gpu=256,  # 32 * 8 = 256
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
    lr=1e-4,
    weight_decay=1e-4, eps=1e-8, betas=(0.9, 0.999),
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
    by_epoch=False, min_lr=1e-6,
)

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

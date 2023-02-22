_base_ = [
    '../../../_base_/datasets/DNA/dna_pretrain.py',
    '../../../_base_/default_runtime.py',
]

embed_dim = 64
seq_len = 512
patch_size = 1

# model settings
model = dict(
    type='BERT',
    pretrained=None,
    mask_ratio=0.15,  # BERT 15%
    backbone=dict(
        type='SimMIMTransformer',
        arch={'embed_dims': embed_dim,
              'num_layers': 12,
              'num_heads': embed_dim // 16,
              'feedforward_channels': embed_dim * 4},
        in_channels=4,
        padding_index=0,
        seq_len=seq_len,
        mask_layer=10,
        mask_ratio=0.15,  # BERT 15%
        mask_token='learnable',
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_rate=0.,  # no dropout for pre-training
        drop_path_rate=0.1,
        final_norm=True,
        out_indices=-1,  # last layer
        with_embedding=True,  # use `nn.Embedding`
        with_cls_token=True,
        output_cls_token=True,
    ),
    neck=dict(
        type='SimMIMNeck', feature_Nd="1d",
        in_channels=embed_dim, out_channels=5, encoder_stride=patch_size),
    head=dict(
        type='MIMHead',
        loss=dict(type='CrossEntropyLoss',
            use_soft=False, use_sigmoid=False, reduction='none', loss_weight=1.0),
        feature_Nd="1d", unmask_weight=0., encoder_in_channels=5,
    ),
    init_cfg=[
        dict(type='TruncNormal', layer=['Conv1d', 'Linear'], std=0.02, bias=0.),
        dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.)
    ],
)

# dataset settings
data_root = 'data/dna/'
data_source_cfg = dict(
    type='DNASeqDataset',
    file_list=None,  # use all splits
    # file_list=["train_0.csv",],  # use all splits
    word_splitor="", data_splitor=",", mapping_name="ACGT",  # gRNA tokenize
    has_labels=True, return_label=False,  # pre-training
    data_type="regression", target_type='total',
    filter_condition=5, max_seq_length=seq_len,
)
data = dict(
    samples_per_gpu=2,  # bs64 x 8gpu x 2 accu = bs1024
    workers_per_gpu=2,
    train=dict(
        data_source=dict(root=data_root+"train", **data_source_cfg)),
)
update_interval = 2  # bs64 x 8gpu x 2 accu = bs1024

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
        'mask_token': dict(weight_decay=0.),
    })

# apex
use_fp16 = False
fp16 = dict(type='mmcv', loss_scale=dict(mode='dynamic'))
optimizer_config = dict(
    grad_clip=dict(max_norm=10.0), update_interval=1)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False, min_lr=1e-5,
    warmup='linear',
    warmup_iters=5, warmup_by_epoch=True,
    warmup_ratio=1e-5,
)

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

_base_ = [
    '../_base_/models/vit_base_deit_p16.py',
    '../_base_/datasets/imagenet_swin_sz224_4xbs256.py',
    '../_base_/default_runtime.py',
]

# model
model = dict(
    backbone=dict(
        use_window=True, init_values=0.1,  # use relative pos encoding (USE_SHARED_RPB) + init value
))

# data
data = dict(samples_per_gpu=256, workers_per_gpu=10)

# interval for accumulate gradient
update_interval = 2  # total: 2 x bs256 x 2 accumulates = bs1024

# optimizer
optimizer = dict(
    type='AdamW',
    lr=2.5e-3 * 1024 / 256,  # 10e-4 for SimMIM
    weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999),
    paramwise_options={
        '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'gamma': dict(weight_decay=0.),  # init value
    },
    constructor='TransformerFinetuneConstructor',
    model_type='vit',
    layer_decay=0.75)

# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=20,  # SimMIM
    warmup_ratio=1e-6,
    warmup_by_epoch=True,
    by_epoch=False)

# apex
use_fp16 = True
fp16 = dict(type='apex', loss_scale=dict(init_scale=512., mode='dynamic'))
# optimizer args
optimizer_config = dict(
    update_interval=update_interval, use_fp16=use_fp16,
    grad_clip=dict(max_norm=5.0),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

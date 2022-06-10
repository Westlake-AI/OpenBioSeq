# dataset settings
data_source_cfg = dict(type='ImageList')
# ImageNet dataset, 100 class
data_train_list = 'data/meta/ImageNet100/train_labeled.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet100/val_labeled.txt'
data_test_root = 'data/ImageNet/val/'

dataset_type = 'ClassificationDataset'
sample_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    dict(type='Resize', size=256, interpolation=3),  # 0.85
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **sample_norm_cfg),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **sample_norm_cfg)])

data = dict(
    samples_per_gpu=64,  # V100: 64 x 4gpus = bs256
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False,
    ))

# validation hook
evaluation = dict(
    initial=False,
    interval=1,
    samples_per_gpu=128,
    workers_per_gpu=4,
    eval_param=dict(
        metric=['accuracy',],
        metric_options=dict(topk=(1,5,), average_mode='macro')
    ),
)

# checkpoint
checkpoint_config = dict(interval=10, max_keep_ckpts=1)

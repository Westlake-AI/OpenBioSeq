# dataset settings
data_source_cfg = dict(type='ImageList')
# Tiny Imagenet
data_train_list = 'data/TinyImageNet/meta/train_labeled.txt'  # train 10w
data_train_root = 'data/TinyImageNet/train/'
data_test_list = 'data/TinyImageNet/meta/val_labeled.txt'  # val 1w
data_test_root = 'data/TinyImageNet/val/'

dataset_type = 'ClassificationDataset'
sample_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=64, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = []
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **sample_norm_cfg)])
test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **sample_norm_cfg)])

data = dict(
    samples_per_gpu=50,  # 50 x 2gpus = 100
    workers_per_gpu=4,
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
    initial=True,
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

# dataset settings
data_root = 'data/Epilepsy/'
data_source_cfg = dict(type='ProcessedDataset', root=data_root)

dataset_type = 'ClassificationDataset'
sample_norm_cfg = dict(mean=[0,], std=[1,])
train_pipeline = [
    dict(type='ToTensor'),
]
test_pipeline = [
    dict(type='ToTensor'),
]
# prefetch
prefetch = False

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(split='test', **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False),
)

# validation hook
evaluation = dict(
    initial=False,
    interval=2,
    samples_per_gpu=128,
    workers_per_gpu=2,
    eval_param=dict(
        metric=['accuracy', 'f1_score',],
        metric_options=dict(topk=(1,), average_mode='macro')
    ),
)

# checkpoint
checkpoint_config = dict(interval=40, max_keep_ckpts=1)

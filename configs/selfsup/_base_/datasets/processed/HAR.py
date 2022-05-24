# dataset settings
data_root = 'data/HAR/'
data_source_cfg = dict(type='ProcessedDataset', return_label=False, root=data_root)

dataset_type = 'MultiViewDataset'
sample_norm_cfg = dict(mean=[0,], std=[1,])
train_pipeline = [
    dict(type='RandomScaling', sigma=1.1, p=1.0),
    dict(type='RandomPermutation', max_segments=8, p=0.5),
    dict(type='RandomJitter', sigma=0.8, p=0.5),
    dict(type='ToTensor'),
]
test_pipeline = [
    dict(type='ToTensor'),
]
# prefetch
prefetch = False

# dataset summary
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', **data_source_cfg),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))

# checkpoint
checkpoint_config = dict(interval=80, max_keep_ckpts=1)

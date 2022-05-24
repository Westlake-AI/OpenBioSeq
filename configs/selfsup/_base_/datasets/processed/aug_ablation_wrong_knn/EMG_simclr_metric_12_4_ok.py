# dataset settings
data_root = 'data/EMG_classification/'
data_source_cfg = dict(type='ProcessedDataset', root=data_root)

dataset_type = 'MultiViewDataset'
sample_norm_cfg = dict(mean=[0,], std=[1,])
train_pipeline = [
    dict(type='RandomScaling', sigma=1.8, p=1.0),
    dict(type='RandomPermutation', max_segments=4, p=1.0),
    dict(type='RandomJitter', sigma=2, p=1.0),
    dict(type='ToTensor'),
]
test_pipeline = [dict(type='ToTensor')]
# prefetch
prefetch = False

# dataset summary
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(split='train', return_label=False, **data_source_cfg),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))

# SSL eval
val_train_pipeline = [dict(type='ToTensor')]
val_test_pipeline = [dict(type='ToTensor')]

val_data = dict(
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type='ClassificationDataset',
        data_source=dict(split='train', return_label=True, **data_source_cfg),
        pipeline=val_train_pipeline,
        prefetch=False,
    ),
    val=dict(
        type='ClassificationDataset',
        data_source=dict(split='test', return_label=True, **data_source_cfg),
        pipeline=val_test_pipeline,
        prefetch=False),
)

# additional hooks
custom_hooks = [
    dict(type='SSLMetricHook',
        val_dataset=val_data['train'],
        train_dataset=val_data['val'],  # remove it if metric_mode is None
        forward_mode='vis',
        metric_mode=['knn', 'svm',],
        # metric_mode=['knn',],
        metric_args=dict(
            knn=20, temperature=0.07, chunk_size=256,
            dataset='onehot', costs_list="0.01,0.1,1.0,10.0", default_cost=None, num_workers=6,),
        # visual_mode='umap',  # 'tsne' or 'umap'
        visual_mode=None,
        visual_args=dict(n_epochs=300, plot_backend='seaborn'),
        save_val=False,  # whether to save results
        initial=True,
        interval=10,
        samples_per_gpu=256,
        workers_per_gpu=4,
        eval_param=dict(topk=(1,))),
]

# checkpoint
checkpoint_config = dict(interval=100, max_keep_ckpts=1)

# dataset settings
data_root = 'data/Epilepsy/'
data_source_cfg = dict(type='ProcessedDataset', root=data_root)

dataset_type = 'ClassificationDataset'
sample_norm_cfg = dict(mean=[0,], std=[1,])
train_pipeline = [dict(type='ToTensor')]
test_pipeline = [dict(type='ToTensor')]
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

# additional hooks
custom_hooks = [
    dict(type='SSLMetricHook',
        val_dataset=data['train'],
        train_dataset=data['val'],  # remove it if metric_mode is None
        forward_mode='vis',
        metric_mode=['knn', 'svm',],
        metric_args=dict(
            knn=20, temperature=0.07, chunk_size=256,
            dataset='onehot', costs_list="0.01,0.1,1.0,10.0", default_cost=None, num_workers=4,),
        visual_mode='umap',  # 'tsne' or 'umap'
        # visual_mode=None,
        visual_args=dict(n_epochs=300, plot_backend='seaborn'),
        save_val=False,  # whether to save results
        initial=True,
        interval=40,
        samples_per_gpu=256,
        workers_per_gpu=4,
        eval_param=dict(topk=(1,))),
]

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

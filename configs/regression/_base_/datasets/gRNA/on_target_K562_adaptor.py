# dataset settings
data_root = 'data/on_target_K562/'
data_source_cfg = dict(
    type='DNASeqDataset',
    file_list=None, k=6, padding_idx=0,
    word_splitor=" ", data_splitor="\t", seq_type='grna',
    data_type="regression", max_seq_length=63
)

dataset_type = 'RegressionDataset'
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
    samples_per_gpu=256,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            root=data_root+"train",
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            root=data_root+"test",
            **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=False),
)

# validation hook
evaluation = dict(
    initial=False,
    interval=2,
    samples_per_gpu=200,
    workers_per_gpu=2,
    eval_param=dict(
        metric=['mse', 'spearman', 'pearson'],
        metric_options=dict(average_mode='mean')
    ),
)

# checkpoint
checkpoint_config = dict(interval=200, max_keep_ckpts=1)

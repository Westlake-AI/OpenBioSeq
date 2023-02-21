# dataset settings
data_root = 'data/dna/'
data_source_cfg = dict(
    type='DNASeqDataset',
    file_list=None,  # use all splits
    word_splitor="", data_splitor=",", mapping_name="ACGT",  # gRNA tokenize
    data_type="regression", target_type='total',
    filter_condition=5, max_seq_length=512
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
    samples_per_gpu=32,
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
    initial=True,
    interval=1,
    samples_per_gpu=100,
    workers_per_gpu=2,
    eval_param=dict(
        metric=['mse', 'spearman'],
        metric_options=dict(average_mode='mean')
    ),
)

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

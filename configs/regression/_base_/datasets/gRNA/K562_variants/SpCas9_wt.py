# dataset settings
data_root = 'data/K562_variants/SpCas9-wt/'
data_source_cfg = dict(
    type='BioSeqDataset',
    file_list=None,  # use all splits
    word_splitor="", data_splitor="\t", mapping_name="ACGT",  # gRNA tokenize
    data_type="regression",
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
    interval=5,
    samples_per_gpu=100,
    workers_per_gpu=2,
    eval_param=dict(
        metric=['mse', 'spearman'],
        metric_options=dict(average_mode='mean')
    ),
)

# checkpoint
checkpoint_config = dict(interval=200, max_keep_ckpts=1)

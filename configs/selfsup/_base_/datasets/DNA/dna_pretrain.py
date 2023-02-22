# dataset settings
data_root = 'data/dna/'
data_source_cfg = dict(
    type='DNASeqDataset',
    file_list=None,  # use all splits
    word_splitor="", data_splitor=",", mapping_name="ACGT",  # gRNA tokenize
    has_labels=True, return_label=False,  # pre-training
    data_type="regression", target_type='total',
    filter_condition=5, max_seq_length=512
)

dataset_type = 'ExtractDataset'
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
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            root=data_root+"train",
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),
)

# checkpoint
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

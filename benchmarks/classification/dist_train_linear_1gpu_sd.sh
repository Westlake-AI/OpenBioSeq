#!/usr/bin/env bash

set -e
set -x

CFG=$1 # use cfgs under "configs/benchmarks/linear_classification/"
PRETRAIN=$2
SD=$3  # random seed
PY_ARGS=${@:4}  # --resume_from --deterministic
GPUS=1 # When changing GPUS, please also change imgs_per_gpu in the config file accordingly to ensure the total batch size is 256.
PORT=${PORT:-29500}

if [ "$CFG" == "" ] || [ "$PRETRAIN" == "" ]; then
    echo "ERROR: Missing arguments."
    exit
fi


WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"


# train
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py \
    $CFG \
    --pretrained $PRETRAIN \
    --work_dir $WORK_DIR --seed $SD --launcher="pytorch" ${PY_ARGS}

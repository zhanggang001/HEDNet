#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

CONFIG=$1
NGPUS=$2
CKPT=$3
PY_ARGS=${@:4}
PROJ_ROOT='/xxx/HEDNet'

export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=${NGPUS} ${PROJ_ROOT}/tools/test.py --launcher pytorch \
    --cfg_file ${CONFIG} --ckpt ${CKPT} --output_dir output --root_dir=${PROJ_ROOT} --dataset='waymo' 2>&1 | tee log.test.$T
    # --dataset='waymo'/'nuscenes'/'argo2'
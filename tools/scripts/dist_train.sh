#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
CONFIG=$1
NGPUS=$2
PY_ARGS=${@:3}
PROJ_ROOT='/xxx/HEDNet'
export PYTHONPATH=${PYTHONPATH}:${PROJ_ROOT}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} \
    ${PROJ_ROOT}/tools/train.py --launcher pytorch --cfg_file ${CONFIG} ${PY_ARGS} \
    --output_dir 'output' --root_dir=${PROJ_ROOT} --sync_bn --eval_map --wo_gpu_stat --dataset='waymo' \
    2>&1 | tee log.train.$T
    # --dataset='waymo'/'nuscenes'/'argo2'

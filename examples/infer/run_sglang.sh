export LD_PRELOAD=/llm_reco_ssd/luoxinchen/libs/libnccl.so.2.21.5.noece.cpu
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_ALGO=^NVLS,NVLSTree
export NCCL_DEBUG=INFO

ADDR=$1
MODEL=$2
TP=$3

export WORLD_SIZE=1
export RANK=0

if [ -n "$OMPI_COMM_WORLD_SIZE" ]; then
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
else
    echo "OMPI_COMM_WORLD_SIZE found. set WORLD_SIZE=$WORLD_SIZE."
fi

if [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    export RANK=$OMPI_COMM_WORLD_RANK
else
    echo "OMPI_COMM_WORLD_RANK found. set WORLD_SIZE=$WORLD_SIZE."
fi


python3 -m sglang.launch_server --model-path $MODEL \
    --tp $TP \
    --dist-init-addr $ADDR \
    --nnodes $WORLD_SIZE \
    --node-rank $RANK \
    --trust-remote-code
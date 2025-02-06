export LD_PRELOAD=/llm_reco_ssd/luoxinchen/libs/libnccl.so.2.21.5.noece.cpu
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_ALGO=^NVLS,NVLSTree
export NCCL_DEBUG=INFO

args="$@"

echo $args

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c \
    "python3 init_servers.py $args"

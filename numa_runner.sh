#!/bin/bash 

local_rank=$OMPI_COMM_WORLD_LOCAL_RANK
local_size=$OMPI_COMM_WORLD_LOCAL_SIZE
global_rank=$OMPI_COMM_WORLD_RANK

num_cpus=$(nproc --all)
num_numa=2
ranks_per_numa=$(($local_size / $num_numa))
numa_id=$(($local_rank / $ranks_per_numa))
rank_inside_numa=$(($local_rank % $ranks_per_numa))
cores_per_rank=$(($num_cpus / $local_size))


numa0_cores=($(numactl -H | grep "node 0 cpus" | awk -F "cpus: " '{print $2}'))
numa1_cores=($(numactl -H | grep "node 1 cpus" | awk -F "cpus: " '{print $2}'))

start=$(($rank_inside_numa * $cores_per_rank))

if [ $numa_id -eq 0 ];then
  cores=${numa0_cores[@]:$start:$cores_per_rank}
  cores=${cores// /,}
elif [ $numa_id -eq 1 ];then
  cores=${numa1_cores[@]:$start:$cores_per_rank}
  cores=${cores// /,}
else
  echo "wrong numa_id"
  exit 1
fi

echo "rank $global_rank bind to numa $numa_id cores [$cores]"

numactl --all -m $numa_id -C $cores $@

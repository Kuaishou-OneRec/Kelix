#!/bin/bash

sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq
hostfile=/etc/mpi/hostfile_seq
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=8  # 总进程数改为8
export HOSTFILE=/etc/mpi/hostfile_seq

# CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置输出文件路径
OUTPUT_PATH="msy_infer_qwen2vl"

# 环境变量设置
KWS_SERVICE_REGION=HB2
KWS_SERVICE_DC=WLF2
KWS_SERVICE_CATALOG=ai-platform.ksnserver.sparse-server
KWS_SERVICE_NAME=ai-platform-mio-kai
KWS_SERVICE_AZ=HB2AZ2
KWS_SERVICE_PAZ=HB2AZ2
KWS_SERVICE_STAGE=PROD
PYTHONPATH=.:$PYTHONPATH

# 通用的MPI参数
COMMON_MPI_PARAMS="--allow-run-as-root -np $np \
    --hostfile /etc/mpi/hostfile_seq \
    --bind-to none \
    --map-by node"

# 通用的环境变量参数
COMMON_ENV_VARS="-x PYTHONPATH=$PYTHONPATH \
    -x JAVA_HOME=$JAVA_HOME \
    -x HIVE_HOME=$HIVE_HOME \
    -x CLASSPATH=$CLASSPATH \
    -x HADOOP_USER_NAME=$HADOOP_USER_NAME \
    -x HADOOP_HOME=$HADOOP_HOME \
    -x SPARK_HOME=$SPARK_HOME \
    -x KWS_SERVICE_REGION=$KWS_SERVICE_REGION \
    -x KWS_SERVICE_DC=$KWS_SERVICE_DC \
    -x KWS_SERVICE_CATALOG=$KWS_SERVICE_CATALOG \
    -x KWS_SERVICE_NAME=$KWS_SERVICE_NAME \
    -x KWS_SERVICE_AZ=$KWS_SERVICE_AZ \
    -x KWS_SERVICE_PAZ=$KWS_SERVICE_PAZ \
    -x KWS_SERVICE_STAGE=$KWS_SERVICE_STAGE"

# 通用的Python脚本参数
COMMON_SCRIPT_PARAMS="--output_path ${OUTPUT_PATH} \
    --batch_size 1"

# 运行单个MPI任务，使用所有GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun $COMMON_MPI_PARAMS $COMMON_ENV_VARS \
    python3 recovlm/benchmark/batch_infer_msy.py \
    $COMMON_SCRIPT_PARAMS

# 等待所有进程完成
wait

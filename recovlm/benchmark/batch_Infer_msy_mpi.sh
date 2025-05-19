#!/bin/bash

hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=1  # Changed from 4 to 1 to match available slots
export HOSTFILE=/etc/mpi/hostfile

# 设置输出文件路径
OUTPUT_PATH="msy_infer.jsonl"

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
    --hostfile /etc/mpi/hostfile \
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
COMMON_SCRIPT_PARAMS="--tp 4 \
    --output_path ${OUTPUT_PATH} \
    --top_p 0.8 \
    --temperature 0.7 \
    --max_tokens 4096 \
    --batch_size 32 \
    --limit_mm_per_prompt 3 \
    --max_frames 10 \
    --num_generations 1"

# 运行MPI任务 (使用所有可用GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun $COMMON_MPI_PARAMS $COMMON_ENV_VARS \
    python3 recovlm/benchmark/batch_infer_msy.py \
    $COMMON_SCRIPT_PARAMS \
    --global_rank 0

# 合并最终结果
cat ${OUTPUT_PATH}.global* > ${OUTPUT_PATH}

# 清理临时文件
rm ${OUTPUT_PATH}.global*
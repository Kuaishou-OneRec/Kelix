#!/usr/bin/env bash
set -e

git config --global user.email 'yangzhuoran@kuaishou.com'
git config --global user.name 'yangzhuoran'

email=$(git config --get user.email)

# 检查 email 是否为空
if [[ -z "$email" ]]; then
        echo "Please set your git email:"
        echo "  git config --global user.email 'you@kuaishou.com'"
        exit 1
else
        echo "Git user.email: $email"
fi

# 这里不需要 hostfile，多机才用，单机直接跳过
# sed 's/=1/=8/g' /etc/mpi/hostfile > /etc/mpi/hostfile_seq

MODEL_DIR=/mmu_mllm_hdd_2/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0714_sp1
OUTPUT_DIR=/llm_reco_ssd/yangzhuoran/code/outputs/navit-tar/test_1_5

mkdir -p $OUTPUT_DIR
mkdir -p /tmp/_wids_cache

# 单机固定为 1
nnode=1

# 注意修改实验内容备注
comment="从53Kstage2初始化，sp1，没有视频"

git add --all
git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR, resume"
git_hash=$(git rev-parse --short HEAD)

set -x

KML_TASK_ID=112233

SCRIPT_FILE=$(readlink -f $0)
echo `date '+%Y-%m-%d %H:%M:%S'` >> $OUTPUT_DIR/task_info.log
echo "task: kml-task-${KML_TASK_ID}-record-${KML_ID}" >> $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "commit_id: ${git_hash}" >> $OUTPUT_DIR/task_info.log
echo "=========================" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH

source set_env.sh

# 单机 1 卡：np=1，回环网卡 lo
MASTER_ADDR=127.0.0.1
MASTER_PORT=8499
np=1
TCP_NIC=lo

nohup mpirun --allow-run-as-root \
        -np $np \ \
        -mca btl self,tcp -mca pml ob1 \
        -mca btl_tcp_if_include $TCP_NIC \
        -mca oob_tcp_if_include $TCP_NIC \
        -mca btl_openib_allow_ib false \
        -mca opal_set_max_sys_limits 1 \
        -x OMPI_MCA_btl=self,tcp \
        -x OMPI_MCA_pml=ob1 \
        -x OMPI_MCA_btl_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_oob_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_btl_openib_allow_ib=false \
        -x NCCL_IB_DISABLE=1 \                       # 🚨 禁止 IB
        -x NCCL_SOCKET_IFNAME=$TCP_NIC \
        -x NCCL_DEBUG=WARN \
        -x LD_PRELOAD=$LD_PRELOAD \
        -x http_proxy="" \
        -x https_proxy="" \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
        -x NO_COLOR=1 \
        -x TERM=dumb \
        -x COLORTERM=0 \
        -x PYTHONIOENCODING=utf-8 \
        -x LD_LIBRARY_PATH=$LIBRARY_PATH \
        -x PATH \
        -x PYTHONPATH=$PYTHONPATH \
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
        -x KWS_SERVICE_STAGE=$KWS_SERVICE_STAGE \
        -x MASTER_ADDR=$MASTER_ADDR \
        -x MASTER_PORT=$MASTER_PORT \
        -x LD_PRELOAD=$LD_PRELOAD \
        -x KAI_FLAG_FILE \
        -x KML_ID \
        -x HADOOP_USER_NAME=$HADOOP_USER_NAME \
        -x TOKENIZERS_PARALLELISM=false \
        -x http_proxy=\
        -x https_proxy=\
        with_nccl_local_env \
        bash -c "bash numa_runner.sh python3 recipes/train_fsdp.py --model_dir $MODEL_DIR \
                --output_dir $OUTPUT_DIR \
                --dataset_config examples/vlm/0.9.7/1d_vs_3d_rope_v2/exp_0.5.11.json \
                --model_class KeyeForConditionalGeneration_vitrope_slowfast \
                --allow_random_init_params 'mlp_AR.pre_norm.weight,mlp_AR.pre_norm.bias,mlp_AR.linear_1.weight,mlp_AR.linear_1.bias,mlp_AR.linear_2.weight,mlp_AR.linear_2.bias,visual_fast.vision_model.embeddings.packing_position_embedding.weight,fast_mlp_AR.pre_norm.weight,fast_mlp_AR.pre_norm.bias,fast_mlp_AR.linear_1.weight,fast_mlp_AR.linear_1.bias,fast_mlp_AR.linear_2.weight,fast_mlp_AR.linear_2.bias' \
                --monitor_datasource_loss \
                --monitor_datasource_cnt \
                --max_length 15500 \
                --learning_rate 2e-5 \
                --vision_learning_rate 2e-6 \
                --min_lr 1e-7 \
                --weight_decay 0.1 \
                --lr_scheduler_type cosine \
                --num_warmup_steps 1000 \
                --num_training_steps 25000 \
                --save_checkpoint_per_step 500 \
                --sequence_parallel_size 1 \
                --use_flash_attention_2 \
                --logging_per_step 20 \
                --fp32_weight \
                --seed 19260817 \
                --enable_gradient_checkpointing \
                --merge_checkpoint \
                --merge_checkpoint_dtype bf16 \
                --merge_checkpoint_output_file pytorch_model.bin \
                --comment '$comment' \
                --commit_id $git_hash \
                --kml_id $KML_ID \
                --kml_task_id $KML_TASK_ID \
                --heartbeat_monitor" > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

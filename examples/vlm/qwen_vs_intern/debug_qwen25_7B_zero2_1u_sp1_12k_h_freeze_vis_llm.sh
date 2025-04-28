
git config --global user.email 'lingzhixin@kuaishou.com'
git config --global user.name 'lingzhixin'

email=$(git config --get user.email)

# 检查 email 是否为空
if [[ -z "$email" ]]; then
        echo "Please set you git email:"
        echo "  git config --global user.email 'you@kuaishou.com'"
        exit 1
else
        echo "Git user.emal: $email"
fi

sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq

# MODEL_DIR=/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage1-v0.0.36/global_step90000-hf
MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-7B-Instruct # Pretrained/Base model path
OUTPUT_DIR=/llm_reco/lingzhixin/exp_outputs/qwen_profile/debug/0.0.1/debug_qwen25_7B_zero2_1u_sp1_12k_h_freeze_vis_llm

mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注
comment="version:0.4.1;model_size:72B;GPU_type:H800;data:inner & outer comments"

git add --all
git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR, resume"
git_hash=$(git rev-parse --short HEAD)

set -x

SCRIPT_FILE=$(readlink -f $0)
echo `date '+%Y-%m-%d %H:%M:%S'` >> $OUTPUT_DIR/task_info.log
echo "task: kml-task-${KML_TASK_ID}-record-${KML_ID}" >> $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "commit_id: ${git_hash}" >> $OUTPUT_DIR/task_info.log
echo "=========================" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH


source set_env.sh

hostfile=/etc/mpi/hostfile_seq
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')

MASTER_ADDR=$MY_NODE_IP
MASTER_PORT=8499

nohup mpirun --allow-run-as-root -np $np \
        -mca plm_rsh_args "-p ${Port}"  \
        -hostfile $hostfile \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
        -bind-to none  -map-by slot \
        -mca opal_set_max_sys_limits 1 \
        -mca plm_rsh_num_concurrent 300 \
        -mca routed_radix 600 \
        -mca btl_tcp_if_include eth04 \
        -mca btl_openib_allow_ib true \
        --mca btl tcp,self \
        -x NO_COLOR=1 \
        -x TERM=dumb \
        -x COLORTERM=0 \
        -x PYTHONIOENCODING=utf-8 \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_NET_OVERHEAD=1000 \
        -x NCCL_PROTO=^LL128 \
        -x NCCL_MIN_NCHANNELS=4 \
        -x NCCL_ALGO=^NVLS,NVLSTree \
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
        -x http_proxy=\
        -x https_proxy=\
        python3 recipes/train_fsdp.py --model_dir $MODEL_DIR \
                --output_dir $OUTPUT_DIR \
                --dataset_config examples/vlm/configs/qwen/qwen_stage3.json \
                --model_processor Qwen2_5_VLProcessor \
                --model_class Qwen2_5_VLForConditionalGeneration \
                --monitor_datasource_loss \
                --monitor_datasource_cnt \
                --max_length 12000 \
                --learning_rate 1e-6 \
                --min_lr 0.0 \
                --weight_decay 0.1 \
                --lr_scheduler_type cosine \
                --num_warmup_steps 500 \
                --num_training_steps 50000 \
                --save_checkpoint_per_step 500 \
                --sequence_parallel_size 1 \
                --use_flash_attention_2 \
                --logging_per_step 10 \
                --fp32_weight \
		--freeze_visual \
		--freeze_llm \
                --seed 19260817 \
                --enable_gradient_checkpointing \
                --merge_checkpoint \
                --merge_checkpoint_dtype bf16 \
                --merge_checkpoint_output_file pytorch_model.bin \
                --comment "$comment" \
                --commit_id $git_hash \
                --kml_id $KML_ID \
                --kml_task_id $KML_TASK_ID \
                --heartbeat_monitor > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &


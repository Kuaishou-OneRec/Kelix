email=$(git config --get user.email)

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "pip3 install timm==1.0.15" 

# 检查 email 是否为空
if [[ -z "$email" ]]; then
        echo "Please set you git email:"
        echo "  git config --global user.email 'you@kuaishou.com'"
        exit 1
else
        echo "Git user.emal: $email"
fi

sed 's/=1/=8/g' /etc/mpi/hostfile  | head -7  > /etc/mpi/hostfile_seq


# MODEL_DIR=/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage1-v0.0.36/global_step90000-hf
MODEL_DIR=/llm_reco_ssd/zhouyang12/models/InternVL3-2Bt/ # Pretrained/Base model path
# MODEL_DIR=/llm_reco/chuchenglong/InternVL/models/OpenGVLab/InternVL2_5-4B
OUTPUT_DIR=/llm_reco_ssd/luoxinchen/output3/RecoVLM-Base/0.7.0/2b/stage_3_v0
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注
comment="run internvl 2b 0.7.0 stage2 by lzx, load from ckpt1k, use stage2-0.6.0 data, note there stage1.5 0.6.0 data is part of stage2 0.6.0 data"


git add --all
git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR"
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
TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")

MASTER_ADDR=$MY_NODE_IP
MASTER_PORT=8499

# debug7b_short.json
# debug7b_fsdp_3p_v1_debug2_orids             
# --enable_gradient_checkpointing \
nohup mpirun --allow-run-as-root \
        -hostfile $hostfile \
        -mca btl self,tcp -mca pml ob1 \
        -mca plm_rsh_num_concurrent 600 \
        -mca routed_radix 600 \
        -mca btl_tcp_if_include $TCP_NIC \
        -mca oob_tcp_if_include $TCP_NIC \
        -mca btl_openib_allow_ib false \
        -mca opal_set_max_sys_limits 1 \
        -x OMPI_MCA_btl=self,tcp \
        -x OMPI_MCA_pml=ob1 \
        -x OMPI_MCA_btl_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_oob_tcp_if_include=$TCP_NIC \
        -x OMPI_MCA_btl_openib_allow_ib=false \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_SOCKET_IFNAME=$TCP_NIC \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_DEBUG=WARN \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_NET_OVERHEAD=1000 \
        -x NCCL_IB_TIMEOUT=20 \
        -x LD_PRELOAD=$LD_PRELOAD \
        -x http_proxy="" \
        -x https_proxy="" \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
        -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
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
        -x http_proxy=\
        -x https_proxy=\
        with_nccl_local_env \
        python3 recipes/train_fsdp.py --model_dir $MODEL_DIR \
                --output_dir $OUTPUT_DIR \
                --monitor_datasource_loss \
                --monitor_datasource_cnt \
                --dataset_config examples/vlm/configs/0.7.0/2b_v0_7_0_internvl_stage3.json  \
                --max_length 21000 \
                --learning_rate 5e-5 \
                --model_class InternVLChatModel \
                --min_lr 0.0 \
                --weight_decay 0.01 \
                --lr_scheduler_type cosine \
                --num_warmup_steps 500 \
                --num_training_steps 10000 \
                --save_checkpoint_per_step 1000 \
                --sequence_parallel_size 1 \
                --use_flash_attention_2 \
                --logging_per_step 10 \
                --fp32_weight \
                --enable_profile \
                --seed 19260817 \
		--monitor_image_tokens \
                --enable_gradient_checkpointing \
                --merge_checkpoint \
                --merge_checkpoint_dtype bf16 \
                --merge_checkpoint_output_file pytorch_model.bin \
                --comment "$comment" \
                --commit_id $git_hash \
		--logging_per_step 10 \
		--resume_from /llm_reco_ssd/luoxinchen/output3/RecoVLM-Base/0.7.0/2b/stage_2_v0/step46000/ \
		--resume_from_tag global_step46000 \
                --kml_id $KML_ID \
                --kml_task_id $KML_TASK_ID \
                --heartbeat_monitor > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &




git config --global user.email 'caojiangxia@kuaishou.com'
git config --global user.name 'caojiangxia'

email=$(git config --get user.email)

# 检查 email 是否为空
if [[ -z "$email" ]]; then
        echo "Please set you git email:"
        echo "  git config --global user.email 'you@kuaishou.com'"
        exit 1
else
        echo "Git user.email: $email"
fi

sed 's/=1/=8/g' /etc/mpi/hostfile > /etc/mpi/hostfile_seq

# MODEL_DIR=/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage1-v0.0.36/global_step90000-hf
MODEL_DIR=/mmu_mllm_hdd_2/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0714
#OUTPUT_DIR=/llm_reco/maosiyang/train_out/0.9.1/keye_2B_stage1/
OUTPUT_DIR=/mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.3/0714/debug1

mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注
comment="version:0.9.3;model_size:8B:H800:512;Stage3:0.5.0,sharevit,15000seqlen"

git add --all
#git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR, resume"
#git_hash=$(git rev-parse --short HEAD)
git_hash=12345

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

hostfile=/etc/mpi/hostfile_seq
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
TCP_NIC=$(ifconfig | grep -B1 " "$(hostname -i)" " | grep -o "^\w*")


MASTER_ADDR=$MY_NODE_IP
MASTER_PORT=8499


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
		--dataset_config /mmu_mllm_hdd_2/lingzhixin/recovlm_data/data/keye_stage3_video_0730_sp1_15k.json \
		--model_class KeyeForConditionalGeneration_vitrope_slowfast \
                --allow_random_init_params 'mlp_AR.pre_norm.weight,mlp_AR.pre_norm.bias,mlp_AR.linear_1.weight,mlp_AR.linear_1.bias,mlp_AR.linear_2.weight,mlp_AR.linear_2.bias,visual_fast.vision_model.embeddings.packing_position_embedding.weight,fast_mlp_AR.pre_norm.weight,fast_mlp_AR.pre_norm.bias,fast_mlp_AR.linear_1.weight,fast_mlp_AR.linear_1.bias,fast_mlp_AR.linear_2.weight,fast_mlp_AR.linear_2.bias' \
                --monitor_datasource_loss \
                --monitor_datasource_cnt \
                --max_length 4000 \
                --learning_rate 1e-10 \
                --vision_learning_rate 1e-10 \
                --min_lr 1e-10 \
                --weight_decay 0.1 \
                --lr_scheduler_type cosine \
                --num_warmup_steps 2100 \
                --num_training_steps 1 \
                --save_checkpoint_per_step 1000 \
                --sequence_parallel_size 1 \
                --use_flash_attention_2 \
                --logging_per_step 1 \
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
		--resume_from /mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.9.3/Stage2/8b/slowfast-0721-0717-v2/step27000 \
		--resume_from_tag global_step27000 \
		--heartbeat_monitor" > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &


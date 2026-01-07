git config --global user.email 'maosiyang@kuaishou.com'
git config --global user.name 'maosiyang'

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
script_name=$(basename "$0" .sh)

# Model and output directories - modify as needed
MODEL_DIR=/mmu_mllm_hdd_2/maosiyang/output/keye_tok_e2e_purevideo/MuseV2/video/stage1/global_step21000/converted

OUTPUT_DIR=/mmu_mllm_hdd_2/maosiyang/output/keye_tok_e2e_purevideo/MuseV2/video/stage2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
mkdir -p $OUTPUT_DIR
KAI_FLAG_FILE=msy
mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注
comment="keye_tokenizer_end2end_image_train_video_stage2"

git add --all
git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR, resume"
git_hash=$(git rev-parse --short HEAD)

set -x

SCRIPT_FILE=$(readlink -f $0)
echo `date '+%Y-%m-%d %H:%M:%S'` >> $OUTPUT_DIR/task_info.log
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
        bash -c "bash numa_runner.sh python3 recipes/train_keye_tok_end2end_video.py \
                --model-dir $MODEL_DIR \
                --output-dir $OUTPUT_DIR \
                --dataset-config examples/keye_tokenizer_end2end_video/run_exp1.6.8_stage2.json \
                --lr 2e-4 \
                --vision_lr 2e-5 \
                --min_lr 1e-7 \
                --weight-decay 0.1 \
                --beta1 0.9 \
                --beta2 0.95 \
                --max-length 16000 \
                --lr-scheduler-type cosine \
                --num-warmup-steps 1000 \
                --logging_per_step 20 \
                --num-training-steps 50000 \
                --save-checkpoint-per-step 1000 \
                --context-parallel-size 1 \
                --use-flash-attention-2 \
                --fp32-weight \
                --codebook_loss_weight 1.0 \
                --commitment_loss_weight 0.25 \
                --seed 19260817 \
                --enable-gradient-checkpointing \
                --prefetch-params-in-forward \
                --comment '$comment' \
                --monitor_datasource_loss \
                --monitor_datasource_cnt \
                --commit-id $git_hash" > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &







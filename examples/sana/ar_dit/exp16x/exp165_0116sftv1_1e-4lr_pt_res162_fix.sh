git config --global user.email 'lingzhixin@kuaishou.com'
git config --global user.name 'lingzhixin'

email=$(git config --get user.email)

# 检查 email 是否为空
if [[ -z "$email" ]]; then
        echo "Please set you git email:"
        echo "  git config --global user.email 'you@kuaishou.com'"
        exit 1
else
        echo "Git user.email: $email"
fi

sed 's/=1/=8/g' /etc/mpi/hostfile | head -99 > /etc/mpi/hostfile_seq
script_name=$(basename "$0" .sh)

PYTHONPATH=. python3 examples/keye_ar/convert_hf_checkpoint.py \
        --hf-checkpoint-path /mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp11/step7000/global_step7000/converted_fix \
        --output-dir /mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp11/step7000/global_step7000/muse_converted_fix

MODEL_DIR=/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px-reproduce-0105/
MODEL_DIR=/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp16x/exp162_0116sftv1_1e-4lr_pt_fix/global_step31000/converted
MODEL_CONFIG=/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px-reproduce-0105/config.json
VAE_DIR=/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/
# IMAGE_TOKENIZER_DIR=/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/
KEYE_AR_DIR=/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp11/step7000/global_step7000/muse_converted_fix

VISUALIZE_DIR=/llm_reco_ssd/zhouyang12/data/val_images/
VISUAL_PARQUET_PATH=/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data0110.parquet

SCRIPT_ABS_PATH=$(readlink -f "$0")
if [ $? -ne 0 ]; then
    # 兼容macOS（macOS无readlink -f，用realpath替代）
    SCRIPT_ABS_PATH=$(realpath "$0")
fi

# 2. 获取脚本所在的目录路径
SCRIPT_DIR=$(dirname "${SCRIPT_ABS_PATH}")

# 3. 提取倒数第二级目录名
SECOND_LAST_DIR=$(basename "$(dirname "${SCRIPT_DIR}")")

# 4. 提取最后一级目录名
LAST_DIR=$(basename "${SCRIPT_DIR}")
SCRIPT_NAME=$(basename "${SCRIPT_ABS_PATH}")
SCRIPT_NAME_NO_SUFFIX=${SCRIPT_NAME%.*}  # 去掉最后一个.及后面的内容
OUTPUT_DIR=/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/${SECOND_LAST_DIR}/${LAST_DIR}/${SCRIPT_NAME_NO_SUFFIX}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注
comment="sana_t2i_pretrain_auto_encoder"

# git add --all
# git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR, resume"
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
        bash -c "python3 recipes/sana/train_sana_ar_dit.py \
                --visualize-parquet-path $VISUAL_PARQUET_PATH \
                --visualize-per-step 500 \
                --keye-ar-dir $KEYE_AR_DIR \
                --num-vis-images 1 \
                --model-dir $MODEL_DIR \
                --vae-dir $VAE_DIR \
                --max-condition-length 720 \
                --output-dir $OUTPUT_DIR \
                --allow-random-init-params "diffusion_connector.0.weight,diffusion_connector.0.bias,diffusion_connector.2.weight,diffusion_connector.2.bias,diffusion_connector.3.weight" \
                --skip-load-params "y_embedder.y_embedding" \
                --dataset-config examples/sana/ar_dit/exp16x/exp165_0116sftv1_1e-4lr_pt_res162_fix.json \
                --resolution-budgets "1024:4" \
                --learning-rate 1e-4 \
                --min-lr 1e-4 \
                --num-decay-steps 10000 \
                --weight-decay 0.0 \
                --image-size 1024 \
                --beta1 0.9 \
                --beta2 0.95 \
                --batch-size 4 \
                --lr-scheduler-type cosine_v2 \
                --num-warmup-steps 100 \
                --num-training-steps 300000 \
                --model-config-overrides model_max_length=720 \
                --condition-on-special-tokens \
                --save-checkpoint-per-step 1000 \
                --logging-per-step 20 \
                --clip-range 9999999 \
                --fp32-weight \
                --fp32-reduce \
                --seed 1917 \
                --global-step 31000 \
                --enable-gradient-checkpointing \
                --prefetch-params-in-forward \
                --enable-profile \
                --condition-on-special-tokens \
                --comment '$comment' \
                --commit-id $git_hash" > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

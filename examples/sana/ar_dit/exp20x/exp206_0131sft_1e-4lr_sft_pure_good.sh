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

# /mmu_mllm_hdd_2/lingzhixin/output/Keye/vqar_11.9.1/v9.15_stage3_0.95_256u_from_v86_fix2/step4000/global_step4000/converted
top_sft_dir=/mmu_mllm_hdd_2/lingzhixin/output/Keye/vqar_11.9.1/sft/v102_sft_1.18.1_24u_from_v86fix2/
step=5000
cd /llm_reco/lingzhixin/recovlm_vlmevalkit/vlmevalkit
PYTHONPATH=. python3 dcp2torch_save.py --dcp_path ${top_sft_dir} --step step${step} --base_model_path /mmu_mllm_hdd_2/zhouyang12/models/onebase_1231_2wtoken/
cd -
PYTHONPATH=. python3 examples/keye_ar/convert_hf_checkpoint.py \
        --hf-checkpoint-path ${top_sft_dir}/step${step}/global_step${step}/converted \
        --output-dir ${top_sft_dir}/step${step}/global_step${step}/muse_converted

cp /mmu_mllm_hdd_2/lingzhixin/output/Keye/vqar_11.9.1_sft/v100_sft_1.17.1_24u/step3500/global_step3500/muse_converted/config.json ${top_sft_dir}/step${step}/global_step${step}/muse_converted/
# PYTHONPATH=. \
# python3 muse/tools/dcp2torch.py /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp16x/exp165_0116sftv1_1e-4lr_pt_res162_fix \
# --source-dir /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp16x/exp165_0116sftv1_1e-4lr_pt_res162_fix/global_step32000/converted \
# --tag global_step80000


# MODEL_DIR=/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px-reproduce-0105/
MODEL_DIR=/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp18x/exp182_0116sftv1_1e-4lr_pt_fix/global_step14000/converted/
MODEL_DIR=/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp18x/exp195_0116sftv1_1e-4lr_pt_fix/global_step18000/converted/

# # 等待MODEL_DIR存在
# # 循环等待，直到目录存在
# while [ ! -d "$MODEL_DIR" ]; do
#     # 打印等待提示信息（包含当前时间，便于排查日志）
#     echo "$(date '+%Y-%m-%d %H:%M:%S'): 目标目录 $MODEL_DIR 尚未存在，将等待3秒后重试..."
#     # 等待3秒钟
#     sleep 3
# done

# # 目录存在后，打印完成提示
# echo "$(date '+%Y-%m-%d %H:%M:%S'): 目标目录 $MODEL_DIR 已存在，等待结束！"



MODEL_CONFIG=/llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px-reproduce-0105/config.json
VAE_DIR=/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/
# IMAGE_TOKENIZER_DIR=/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/
KEYE_AR_DIR=${top_sft_dir}/step${step}/global_step${step}/muse_converted
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
                --num-vis-images 14 \
                --model-dir $MODEL_DIR \
                --vae-dir $VAE_DIR \
                --max-condition-length 720 \
                --output-dir $OUTPUT_DIR \
                --allow-random-init-params "diffusion_connector.0.weight,diffusion_connector.0.bias,diffusion_connector.2.weight,diffusion_connector.2.bias,diffusion_connector.3.weight" \
                --skip-load-params "y_embedder.y_embedding" \
                --dataset-config examples/sana/ar_dit/exp20x/exp201_0131sft_1e-4lr_sft_pure.json\
                --resolution-budgets "1024:8" \
                --learning-rate 2e-4 \
                --min-lr 5e-5 \
                --num-decay-steps 8000 \
                --weight-decay 0.0 \
                --image-size 1024 \
                --beta1 0.9 \
                --beta2 0.95 \
                --batch-size 8 \
                --lr-scheduler-type cosine_v2 \
                --num-warmup-steps 200 \
                --num-training-steps 100000 \
                --model-config-overrides model_max_length=720 \
                --condition-on-special-tokens \
                --save-checkpoint-per-step 1000 \
                --logging-per-step 20 \
                --clip-range 9999999 \
                --fp32-weight \
                --fp32-reduce \
                --seed 1917 \
                --global-step 0 \
                --enable-gradient-checkpointing \
                --prefetch-params-in-forward \
                --enable-profile \
                --condition-on-special-tokens \
                --comment '$comment' \
                --commit-id $git_hash" > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

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
MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct # Pretrained/Base model path
OUTPUT_DIR=/llm_reco_ssd/luoxinchen/output3/RecoVLM-Base/0.4.1/cmt_72B

mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

export LD_PRELOAD=/llm_reco_ssd/luoxinchen/libs/libnccl.so.2.21.5.noece.cpu
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_ALGO=^NVLS,NVLSTree


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

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/train_fsdp.py \
     --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_config /llm_reco/maosiyang/msy_m_pt/recovlm/examples/vlm/configs/msy_stage1_cmt_7b_2.json \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --load_weights_only \
    --auto_resume_local_latest \
    --enable_gradient_checkpointing \
    --max_length 30000 \
    --learning_rate 1e-6 \
    --min_lr 0.0 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 500 \
    --num_training_steps 54000 \
    --save_checkpoint_per_step 1000 \
    --sequence_parallel_size 4 \
    --use_flash_attention_2 \
    --reshard_after_forward true \
    --logging_per_step 10 \
    --seed 19260817 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
    --comment "$comment" \
    --commit_id $git_hash \
    --kml_id $KML_ID \
    --kml_task_id $KML_TASK_ID \
    --deepspeed --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &


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

# MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-Qwen2VL-7B-vit # Instructed model path
MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Qwen2VL-7B-vit # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/luoxinchen/output2/RecoVLM/Qwen2-VL-7B-stage1/0.0.42 # 

mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

comment="一阶段训练，load qwenvl vit，只训练adapter"

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

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/pretrain_vl.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_config examples/vlm/configs/stage1_laion_stg1.json \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --max_length 4096 \
    --learning_rate 5e-4 \
    --min_lr 5e-5 \
    --auto_resume_local_latest \
    --lr_scheduler_type cosine \
    --num_warmup_steps 1000 \
    --num_training_steps 15000 \
    --save_checkpoint_per_step 1000 \
    --use_flash_attention_2 \
    --logging_per_step 10 \
    --freeze_llm \
    --freeze_visual_without_adapter \
    --seed 19260817 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
    --comment "$comment" \
    --commit_id $git_hash \
    --kml_id $KML_ID \
    --kml_task_id $KML_TASK_ID \
    --deepspeed --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

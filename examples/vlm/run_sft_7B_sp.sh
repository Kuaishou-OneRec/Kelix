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
MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14 # Pretrained/Base model path
OUTPUT_DIR=/llm_reco_ssd/luoxinchen/output2/RecoVLM/Qwen2-VL-7B-sft/0.0.1

mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注
comment="sft，warmup0.0.25.3-step36000，only llava_one_vision dataset，seq=32k"

git add --all
git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR"
git_hash=$(git rev-parse --short HEAD)

set -x

SCRIPT_FILE=$(readlink -f $0)
echo "task: kml-task-${KML_TASK_ID}-record-${KML_ID}" > $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "commit_id: ${git_hash}" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/pretrain_vl.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_config ./examples/vlm/configs/sft_llava_one_vision.json \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --resume_from /llm_reco_ssd/luoxinchen/output2/RecoVLM/Qwen2-VL-7B-stage2/0.0.25.3 \
    --resume_from_tag global_step36000 \
    --load_weights_only \
    --auto_resume_local_latest \
    --enable_gradient_checkpointing \
    --max_length 32768 \
    --learning_rate 1e-6 \
    --min_lr 0.0 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 500 \
    --num_training_steps 70000 \
    --save_checkpoint_per_step 2000 \
    --sequence_parallel_size 4 \
    --use_flash_attention_2 \
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

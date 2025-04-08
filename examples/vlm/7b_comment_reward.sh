git config --global user.email 'zangdunju@kuaishou.com'
git config --global user.name 'zangdunju'

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
MODEL_DIR=/llm_reco_ssd/luoxinchen/output3/RecoVLM-Base/0.3.1/cmt/global_step9001/merged9001 # Pretrained/Base model path
OUTPUT_DIR=/llm_reco_ssd/zangdunju/output2/RecoVLM/Qwen2-VL-7B-RL/reward_model/0.0.0.4_sample

mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注
comment="sft 0.0.0.1: stg2 is 0.0.0.1 token reward model after sft"

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



# --resume_from /llm_reco_ssd/luoxinchen/output2/RecoVLM/Qwen2-VL-7B-stage2/0.0.25.5 \
# --resume_from_tag global_step32000 \
# --auto_resume_local_latest \
# --load_weights_only \



nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/rlhf.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_config /llm_reco/zangdunju/vllm/rlhf/recovlm/examples/vlm/configs/comment_reward.json \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --enable_gradient_checkpointing \
    --max_length 32768 \
    --learning_rate 1e-6 \
    --min_lr 0.0 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 200 \
    --num_training_steps 2000 \
    --save_checkpoint_per_step 400 \
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
    --load_weights_only \
    --loss_style sample \
    --deepspeed \
   --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &

    # --resume_from_tag global_step1600 \
    # --resume_from /llm_reco_ssd/lingzhixin/model_output_vvcmp/RecoVLM/Qwen2-VL-7B-sft_good_ids1_use_cot1/0.0.0.1/ \

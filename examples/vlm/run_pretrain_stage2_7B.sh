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

OUTPUT_DIR=/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage2/0.0.20.7


mkdir -p $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

# 注意修改实验内容备注

comment="the_cauldron_recaption,32k"

git add --all
git commit -m "email=$email,time=$(date +"%Y%m%d %H:%M:%S"),script=$0,node=$nnode,comment=$comment,output=$OUTPUT_DIR"
git_hash=$(git rev-parse --short HEAD)

set -x

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/pretrain_vl.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --monitor_datasource_loss \
    --monitor_datasource_cnt \
    --dataset_config examples/vlm/configs/the_cauldron_recaption.json \
    --resume_from /llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage1-v0.0.36 \
    --resume_from_tag global_step90000 \
    --load_weights_only \
    --enable_gradient_checkpointing \
    --max_length 32000 \
    --load_weights_only \
    --learning_rate 5e-5 \
    --min_lr 1e-6 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 500 \
    --num_training_steps 40000 \
    --save_checkpoint_per_step 2000 \
    --sequence_parallel_size 4 \
    --use_flash_attention_2 \
    --logging_per_step 1 \
    --seed 19260817 \
    --freeze_llm \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
    --monitor_datasource_cnt \
    --comment "$comment" \
    --commit_id $git_hash \
    --deepspeed --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json >> $OUTPUT_DIR/stdout.log 2>>$OUTPUT_DIR/stderr.log &

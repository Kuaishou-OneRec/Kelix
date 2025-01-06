sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq

MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14 # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/zhangzixing/output/RecoVLM/Qwen2-VL-debug

mkdir $OUTPUT_DIR

mkdir -p /tmp/_wids_cache

nnode=$(wc -l < /etc/mpi/hostfile_seq)

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/pretrain_vl.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset /llm_reco_ssd/luoxinchen/dataset/kwai_video_caption/20250105/index.json \
    --max_length 3072 \
    --learning_rate 2e-4 \
    --min_lr 1e-6 \
    --lr_scheduler_type cosine_with_min_lr \
    --num_warmup_steps 1000 \
    --num_training_steps 90000 \
    --save_checkpoint_per_step 3000 \
    --use_flash_attention_2 \
    --freeze_llm \
    --data_format chatml \
    --logging_per_step 1 \
    --seed 19260817 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
	--deepspeed --deepspeed_config examples/sft/configs/ds_z2_config_7B.json >> $OUTPUT_DIR/stdout.log 2>>$OUTPUT_DIR/stderr.log &

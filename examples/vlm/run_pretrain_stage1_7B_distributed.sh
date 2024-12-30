sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq

MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14 # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct-LLaVA-CC3M-Pretrain-595K-dist

nnode=$(wc -l < /etc/mpi/hostfile_seq)

echo "Output: $OUTPUT_DIR"

deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
	pretrain_stage1.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset /llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K \
    --batch_size 8 \
    --max_length 1536 \
    --save_checkpoint_every_epoch \
    --num_epochs 1 \
    --logging_per_step 1 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
	--deepspeed --deepspeed_config configs/ds_config.json

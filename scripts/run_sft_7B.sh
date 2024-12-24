sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq

MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/zhouyang12/output/NegativeFeedback/models/i2i/qwen2_7B/ds

nnode=$(wc -l < /etc/mpi/hostfile_seq)

echo "Output: $OUTPUT_DIR"

deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
	finetune.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset /llm_reco_ssd/zhouyang12/data/NegativeFeedback/i2i/pairwise/eval/chat_hop2_20k_mm_tmp_train_openrlhf.jsonl \
    --batch_size 8 \
    --max_length 2048 \
    --save_checkpoint_every_epoch \
    --num_epochs 1 \
    --logging_per_step 1 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
	--deepspeed --deepspeed_config configs/ds_config.json

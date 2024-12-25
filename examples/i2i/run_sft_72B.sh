sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq

MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-72B-Instruct # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/zhouyang12/output/NegativeFeedback/models/i2i/qwen2_72B/ds

nnode=$(wc -l < /etc/mpi/hostfile_seq)

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=/llm_reco_ssd/zhouyang12/code/RecoVLM:$PYTHONPATH

deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
	recipes/finetune.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset /llm_reco_ssd/zhouyang12/data/NegativeFeedback/i2i/pairwise/eval/chat_hop2_20k_mm_tmp_train_openrlhf.jsonl \
    --chat_template chat_template_with_generation_tag \
    --input_key conversations \
    --system_prompt /llm_reco_ssd/zhouyang12/code/RecoVLM/examples/i2i/prompts/pairwise.txt \
    --enable_gradient_checkpointing \
    --max_length 2048 \
    --save_checkpoint_every_epoch \
    --use_flash_attention_2 \
    --num_epochs 1 \
    --logging_per_step 1 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
	--deepspeed --deepspeed_config examples/i2i/configs/ds_z3_config_72B.json

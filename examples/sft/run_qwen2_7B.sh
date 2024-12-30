sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq

MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-OpenHermes2_5

nnode=$(wc -l < /etc/mpi/hostfile_seq)

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=/llm_reco_ssd/zhouyang12/code/RecoVLM:$PYTHONPATH

#     --use_flash_attention_2 \

#   --enable_gradient_checkpointing \
deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
	recipes/finetune.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset /llm_reco_ssd/luoxinchen/dataset/OpenHermes-2.5/openhermes2_5.json \
    --chat_template chat_template_with_generation_tag \
    --input_key conversations \
    --role_key from \
    --content_key value \
    --user_name human \
    --assistant_name gpt \
    --file_format json \
    --max_length 2048 \
    --enable_gradient_checkpointing \
    --save_checkpoint_every_epoch \
    --num_epochs 1 \
    --logging_per_step 1 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
	--deepspeed --deepspeed_config examples/sft/configs/ds_z3_config_72B.json

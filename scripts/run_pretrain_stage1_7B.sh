MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14 # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct-LLaVA-CC3M-Pretrain-595K

rm -rf __pycache__
rm -rf $OUTPUT_DIR

echo $OUTPUT_DIR

deepspeed --hostfile=hostfile --num_nodes=1 --num_gpus=8 \
	pretrain_stage1.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset /llm_reco_ssd/luoxinchen/dataset/LLaVA-CC3M-Pretrain-595K \
    --batch_size 1 \
    --max_length 4096 \
    --save_checkpoint_every_epoch \
    --num_epochs 2 \
    --logging_per_step 1 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
	--deepspeed --deepspeed_config configs/ds_config.json
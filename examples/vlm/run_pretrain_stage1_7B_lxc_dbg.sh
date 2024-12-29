sed 's/=1/=8/g' /etc/mpi/hostfile  | head -1000 > /etc/mpi/hostfile_seq

MODEL_DIR=/llm_reco_ssd/zhouyang12/models/Qwen2-7B-Instruct-DFN5B-ViT-H-14 # Pretrained model path
OUTPUT_DIR=/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage1-v0.0.2

nnode=$(wc -l < /etc/mpi/hostfile_seq)

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH


#   --enable_gradient_checkpointing \
deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
	recipes/pretrain_vl.py --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset /llm_reco_ssd/luoxinchen/dataset/coyo-700m-webdataset/coyo-700m-index.json \
    --max_length 384 \
    --save_checkpoint_every_epoch \
    --packing_batch_size 4 \
    --use_flash_attention_2 \
    --freeze_llm \
    --num_epochs 1 \
    --logging_per_step 1 \
    --merge_checkpoint \
    --merge_checkpoint_dtype bf16 \
    --merge_checkpoint_output_file pytorch_model.bin \
	--deepspeed --deepspeed_config examples/sft/configs/ds_z3_config_7B.json

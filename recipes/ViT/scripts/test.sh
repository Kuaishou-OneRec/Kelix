
export PYTHONPATH=$PWD:$PYTHONPATH

nnode=$(wc -l < /etc/mpi/hostfile_seq)

OUTPUT_DIR=/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/0.0.0.1

echo $OUTPUT_DIR

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/ViT/training/trainer/siglip.py \
    --config_file /llm_reco/zangdunju/vllm/rlhf/recovlm/recipes/ViT/configs/v1.yaml \
    --output_dir $OUTPUT_DIR \
    --deepspeed \
    --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &
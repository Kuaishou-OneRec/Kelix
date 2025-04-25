
export PYTHONPATH=$PWD:$PYTHONPATH

nnode=$(wc -l < /etc/mpi/hostfile_seq)

CONFIG_FILE=/llm_reco/zangdunju/vllm/vit/recovlm/recipes/ViT/configs/v1.yaml

OUTPUT_DIR=/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/0.0.0.15

echo $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/ViT/training/trainer/siglip.py \
    --config_file $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --deepspeed \
    --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log &
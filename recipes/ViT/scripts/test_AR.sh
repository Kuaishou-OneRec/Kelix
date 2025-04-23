
export PYTHONPATH=$PWD:$PYTHONPATH

nnode=$(wc -l < /etc/mpi/hostfile_seq)

OUTPUT_DIR=/llm_reco/liuyang76/logs/vit_train_logs/siilip/0.0.0.1

echo $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/ViT/training/trainer/siglip.py \
    --config_file /llm_reco/liuyang76/codes/llm_rec_code_base/recovlm/recipes/ViT/configs/v1.yaml \
    --output_dir $OUTPUT_DIR \
    --deepspeed \
    --deepspeed_config examples/vlm/configs/ds_stage2.json > $OUTPUT_DIR/stdout.log 2>$OUTPUT_DIR/stderr.log
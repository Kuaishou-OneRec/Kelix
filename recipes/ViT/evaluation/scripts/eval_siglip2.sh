nnode=$(wc -l < /etc/mpi/hostfile_seq)

CONFIG_FILE=/llm_reco/zangdunju/vllm/vit/recovlm/recipes/ViT/evaluation/configs/siglip.yaml

EVAL_DIR=/llm_reco_ssd/zangdunju/output2/RecoVLM/Eval/SigLIP/3.0.0.1_temp
CKPT_FOLDER=/llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/3.0.0.1/global_step10000

echo $EVAL_DIR

mkdir -p $EVAL_DIR

nohup deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/ViT/evaluation/eval_siglip2.py \
    --config_file $CONFIG_FILE \
    --eval_dir $EVAL_DIR \
    --ckpt_folder $CKPT_FOLDER \
    --deepspeed \
    --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json > $EVAL_DIR/stdout.log 2>$EVAL_DIR/stderr.log &


export PYTHONPATH=$PWD:$PYTHONPATH

sed 's/=1/=8/g' /etc/mpi/hostfile  | head -100 > /etc/mpi/hostfile_seq

nnode=$(wc -l < /etc/mpi/hostfile_seq)

OUTPUT_DIR=/llm_reco_ssd/caojiangxia/output2/RecoVLM/SigLIP/100.0.0.7

echo $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

deepspeed --hostfile=/etc/mpi/hostfile_seq --num_nodes=$nnode \
    recipes/ViT/evaluation/eval.py \
    --config_file recipes/ViT/configs/eval.yaml \
    --output_dir $OUTPUT_DIR \
    --deepspeed \
    --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json

# deepspeed --num_nodes=1 \
#     recipes/ViT/evaluation/coco_contrastive_deepspeed.py \
#     --config_file recipes/ViT/configs/v1.yaml \
#     --output_dir $OUTPUT_DIR \
#     --deepspeed \
#     --deepspeed_config examples/vlm/configs/ds_z1_config_7B.json


mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile_seq --pernode bash -c "ps -ef | grep -v '<defunct>' | grep 'recipes/ViT/evaluation/imagenet_zero_shot_deepspeed.py' | awk '{print $2}' | xargs kill -9"
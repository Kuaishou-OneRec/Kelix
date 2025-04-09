#!/bin/bash

MODEL_DIR=${1:-"/llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct"}
DATASET_CONFIG=${2:-"/llm_reco/zhouyang12/.cache/Dataset/KwaiCommentJudgeCot/dataset_config.json"}
#OUTPUT_DIR=${3:-"/llm_reco_ssd/luoxinchen/output3/RecoVLM-Base/0.3.1/cmt/global_step9001/merged9001/bot_comment_v2"}
OUTPUT_DIR=/llm_reco/zhouyang12/.cache/Dataset/KwaiCommentJudgeCot/Qwen2-VL-72B-Instruct

mkdir -p $OUTPUT_DIR 

bash tools/init_ray_cluster.sh

ray job submit --working-dir ./ -- \
    python3 recipes/offline_batch_inference.py \
    --model_dir $MODEL_DIR \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT_DIR \
    --num_workers 0 \
    --num_gpus_per_node 8 \
    --num_inference_node 1 \
    --tp_size 8 \
    --num_generations 1 \
    --max_new_tokens 8192 \
    --batch_size 256 \
    --temperature 0.01\
    --top_p 0.95 \
    --repetition_penalty 1.05 \
    --limit_mm_per_prompt 30 \
    --use_tqdm

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "ray stop"
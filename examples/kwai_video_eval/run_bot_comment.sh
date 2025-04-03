#!/bin/bash

MODEL_DIR=${1:-"/llm_reco_ssd/luoxinchen/output3/RecoVLM-Base/0.3.1/cmt/global_step9001/merged9001"}
DATASET_CONFIG=${2:-"examples/kwai_video_eval/config.json"}
OUTPUT_DIR=${3:-"examples/kwai_video_eval/output"}

bash tools/init_ray_cluster.sh

ray job submit --working-dir ./ -- \
    python3 recipes/offline_batch_inference.py \
    --model_dir $MODEL_DIR \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT_DIR \
    --num_workers 8 \
    --num_gpus_per_node 8 \
    --num_inference_node 1 \
    --tp_size 4 \
    --num_generations 1 \
    --max_new_tokens 2048 \
    --batch_size 128 \
    --temperature 1.0 \
    --top_p 0.95 \
    --repetition_penalty 1.05 \
    --limit_mm_per_prompt 10

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "ray stop"
#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_dir> <dataset_config> <output_dir> [num_workers] [num_gpus_per_node] [num_inference_node] [tp_size]"
    echo "Example: $0 /path/to/model examples/doc2latex/config.json /path/to/output 8 8 4 8"
    exit 1
fi

MODEL_DIR=$1
DATASET_CONFIG=$2
OUTPUT_DIR=$3
NUM_WORKERS=${4:-8}           # Default value: 8
NUM_GPUS_PER_NODE=${5:-8}     # Default value: 8
NUM_INFERENCE_NODE=${6:-2}    # Default value: 4
TP_SIZE=${7:-4}               # Default value: 8

bash tools/init_ray_cluster.sh

ray job submit --working-dir ./ -- \
    python3 recipes/offline_batch_inference.py \
    --model_dir $MODEL_DIR \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT_DIR \
    --num_workers $NUM_WORKERS \
    --num_gpus_per_node $NUM_GPUS_PER_NODE \
    --num_inference_node $NUM_INFERENCE_NODE \
    --tp_size $TP_SIZE \
    --num_generations 5 \
    --max_new_tokens 8192 \
    --batch_size 256 \
    --temperature 1.0 \
    --top_p 0.95 \
    --repetition_penalty 1.05 \
    --limit_mm_per_prompt 10

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "ray stop"
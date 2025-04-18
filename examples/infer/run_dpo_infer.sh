#!/bin/bash

bash tools/init_ray_cluster.sh

ray job submit --working-dir ./ -- \
    python3 examples/infer/dpo_infer.py \
    --model_dir /llm_reco/chuchenglong/DPO/output/v1-20250408-003739/checkpoint-273-merged \
    --output_dir="/llm_reco/zangdunju/dataset/reward/machine/v2" \
    --tp_size 4 \
    --num_nodes 6 \
    --num_gpus_per_node 8 \
    --limit_mm_per_prompt 10 \
    --top_p 0.95 \
    --temperature 1.0 \
    --max_tokens 4096 \
    --epochs 1 \
    --save_nrows 510 \
    --num_workers 0 \
    --file_list /llm_reco_ssd/zangdunju/dataset/hdfs_data/comment_reward_fs.json \
    --repetition_penalty 1.2 \
    --batch_size 8 \

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "ray stop"
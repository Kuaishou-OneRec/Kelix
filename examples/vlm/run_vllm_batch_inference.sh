
bash tools/init_ray_cluster.sh

ray job submit --working-dir ./ -- \
    python3 recipes/offline_batch_inference.py \
    --model_dir /llm_reco_ssd/zhouyang12/models/Qwen2-VL-72B-Instruct \
    --dataset_config examples/vlm/configs/vllm_math_360k.json \
    --output_dir /llm_reco_ssd/zhouyang12/data/math_5w \
    --num_workers 8 \
    --num_gpus_per_node 8 \
    --num_inference_node 17 \
    --tp_size 8 \
    --num_generations 1 \
    --max_new_tokens 1024 \
    --batch_size 1024 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 50 \
    --repetition_penalty 1.02 \
    --limit_mm_per_prompt 10

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "ray stop"
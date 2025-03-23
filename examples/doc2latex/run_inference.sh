
bash tools/init_ray_cluster.sh

ray job submit --working-dir ./ -- \
    python3 recipes/offline_batch_inference.py \
    --model_dir /llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-72B-Instruct \
    --dataset_config examples/doc2latex/config.json \
    --output_dir /mmu_nlp_hdd/zhouyang12/data/pdf2png_process/output \
    --num_workers 8 \
    --num_gpus_per_node 8 \
    --num_inference_node 4 \
    --tp_size 8 \
    --num_generations 1 \
    --max_new_tokens 8192 \
    --batch_size 1024 \
    --temperature 1.0 \
    --top_p 0.001 \
    --top_k 1 \
    --repetition_penalty 1.05 \
    --limit_mm_per_prompt 10

mpirun --allow-run-as-root --hostfile /etc/mpi/hostfile --pernode bash -c "ray stop"
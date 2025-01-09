export NCCL_CUMEM_ENABLE=0
export RAY_memory_monitor_refresh_ms=0
export NCCL_P2P_DISABLE=1
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=TRACE
export VLLM_TRACE_FUNCTION=1

python3 -u ray_batch_infer.py \
--model_folder="/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage2/0.0.12/" \
--model_name="stage2_12" \
--logging_folder="/llm_reco_ssd/luoxinchen/output/RecoVLM/debug_stg2_v1/log/test/" \
--infer_chekpoint_file="/llm_reco_ssd/luoxinchen/output/RecoVLM/debug_stg2_v1/infer_checkpoint_test.txt" \
--MMMU_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMMU/" \
--VideoMME_path="/reco-root/krp/pub/wangqianqian/video_llm/infer_benchmark/dataset/video_mme/video_mme_hetu_format.json" \
--MMBenchEN_path="/reco-root/krp/pub/wangqianqian/video_llm/infer_benchmark/MMMU_infer/dataset/MMBench/en/dev-00000-of-00001.parquet" \
--output_path="/llm_reco_ssd/luoxinchen/output/RecoVLM/debug_stg2_v1/MMMU_infer_test/" \
--benchmark_original_data="/reco-root/krp/pub/wangqianqian/video_llm/infer_benchmark/MMMU_infer/dataset/MMBench/en/dev-00000-of-00001.parquet" 

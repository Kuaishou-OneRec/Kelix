#vllm serve /llm_reco_ssd/zhouyang12/models/Qwen2-VL-7B-Instruct/ --api-key token-abc123 --tensor-parallel-size 4
python3 -u batch_infer.py \
--model_name_or_path="/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage1-v0.0.36/global_step135000" \
--input_path="/reco-root/krp/pub/wangqianqian/video_llm/infer_benchmark/MMMU_infer/dataset/MMMU/" \
--output_path="/reco-root/krp/pub/wangqianqian/video_llm/infer_benchmark/MMMU_infer/eval/MMMU/infer_results/" \

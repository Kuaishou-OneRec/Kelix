model="/llm_reco_ssd/luoxinchen/output/RecoVLM/Qwen2-VL-7B-stage1-v0.0.36"

for val in MMMU MMBenchEN MMBenchCN MME MMTBench MMStar MathVista OCRBench Flickr30k AI2D AI2D_no_mask InfoVQA RealWorldQA
do
        python3 -u ray_batch_infer_new_1.py \
                --MMMU_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMMU/" \
                --VideoMME_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/VideoMME/video_mme_hetu_format.json" \
                --MMBenchEN_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet" \
                --MMBenchCN_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/cn/dev-00000-of-00001.parquet" \
                --ChartQA_path="/hetu_group/wenbin/mllm/benchmark/VQA_Power/ChartQA.json" \
                --MME_path="/hetu_group/huyifei/work_dir/20230509_MLLM_Benchmark/7.MLLM_Benchmark_dataset/MME/MME.json" \
                --MMTBench_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMTBench/mmt_bench_485_hetu_format.json" \
                --MMStar_path="/mmu_mllm_hdd/shiyaya/dataset/mm_reasoning/benchmark/MMStar/YuanQi/mmstar.json" \
                --MathVista_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MathVista/mathvista.json" \
                --OCRBench_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/OCRBench/data/test-00000-of-00001.parquet" \
                --Flickr30k_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/flickr30k/flickr30k_karpathy_test.json" \
                --AI2D_path="/llm_reco_ssd/luoxinchen/dataset/ai2d/ai2d/data/merge/test-00000-of-00001.parquet" \
                --AI2D_no_mask_path="/llm_reco_ssd/luoxinchen/dataset/ai2d/ai2d-no-mask/data/merge/test-00000-of-00001.parquet" \
                --InfoVQA_path="/llm_reco_ssd/luoxinchen/dataset/infoVQA/human_download/infographicsvqa_qas/reconstruct_val.json" \
                --RealWorldQA_path="/llm_reco/luoxinchen/dataset/RealWorldQA/RealWorldQA/data/merge/test-00000-of-00001.parquet" \
                --mmbenchEn_benchmark_original_data="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet" \
		--mmbenchCn_benchmark_original_data="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/cn/dev-00000-of-00001.parquet" \
                --model_folder=$model \
                --logging_folder="${model}/log/test1/" \
                --output_path="${model}/test1/" \
                --infer_${val}=1 \
                --${val}_infer_chekpoint_file="${model}/${val}_infer_checkpoint_greedy_test_1.txt" 
        wait
done

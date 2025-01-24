machines=$(cat /etc/mpi/hostfile | wc -l)
slots=8
echo "machines:"
echo $machines

while true
do
  while IFS= read -r model; do
    if [ -z "$model" ]
    then
      :
    else
      #for val in MMMU MMBenchEN MMBenchCN MME MMTBench MMStar MathVista OCRBench Flickr30k Benchmark_v21 VideoMME
      for val in MME
      do
        python3 -u ray_batch_infer.py \
		--GPU_num=$(($slots*$machines)) \
                --MMMU_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMMU/" \
                --VideoMME_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/VideoMME/video_mme_hetu_format.json" \
                --MMBenchEN_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet" \
                --MMBenchCN_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/cn/dev-00000-of-00001.parquet" \
                --MME_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MME/MME.json" \
                --MMTBench_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMTBench/mmt_bench_485_hetu_format.json" \
                --MMStar_path="/mmu_mllm_hdd/shiyaya/dataset/mm_reasoning/benchmark/MMStar/YuanQi/mmstar.json" \
                --MathVista_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MathVista/mathvista.json" \
                --OCRBench_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/OCRBench/data/test-00000-of-00001.parquet" \
                --Flickr30k_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/flickr30k/flickr30k_karpathy_test.json" \
		--Benchmark_v21_path="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/Benchmark_v21/Benchmark_v21.json" \
                --mmbenchEn_benchmark_original_data="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet" \
                --mmbenchCn_benchmark_original_data="/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/cn/dev-00000-of-00001.parquet" \
                --infer_steps=1 \
                --model_folder=$model \
                --logging_folder="${model}/log/benchmark/" \
                --output_path="${model}/benchmark_output/" \
                --infer_${val}=1 \
                --${val}_infer_chekpoint_file="${model}/${val}_infer_checkpoint.txt"
        wait
      done
    fi
  done < monitor_models.conf
done

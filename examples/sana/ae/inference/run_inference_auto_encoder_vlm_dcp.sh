
# 注意: max-condition-length要按照实际vlm输出的token数来，因为计算rope时要获取实际的height和width，如果max-condition-length过大，会导致rope的计算与实际token数有diff
# /mmu_mllm_hdd_2/zhouyang12/output/MuseV2/sana_v2/multi_scale/exp3.22
torchrun --nproc_per_node=8 recipes/sana/inference_auto_encoder_vlm.py \
        --model-dir /mmu_mllm_hdd_2/zangdunju/output/MuseV2_2e-5/sana/ar_dit/exp100_run_ar_dit_1024_324tokens_reproduce/global_step51000/converted \
        --dcp-dir /mmu_mllm_hdd_2/zhouyang12/output/MuseV2/sana_v2/multi_scale/exp3.22/ \
        --dcp-tag global_step100000 \
        --vae-dir /llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/ \
        --image-tokenizer-dir /llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/ \
        --input-dir /mmu_mllm_hdd_2/zangdunju/muse/mtp/infer/muse/0.7.58/18000step/greedy/GenEval \
        --output-dir /mmu_mllm_hdd_2/zangdunju/code/dev/muse_v2/exp3.22/0.7.58/18000step/dit20000_1024/greedy/GenEval --image-size 1024 --num-generation-images 4 --max-condition-length 324

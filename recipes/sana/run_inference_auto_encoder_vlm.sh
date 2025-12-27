
torchrun --nproc_per_node=8 recipes/sana/inference_auto_encoder_vlm.py \
        --model-dir /mmu_mllm_hdd_2/zhouyang12/output/MuseV2/sana_v2/multi_scale/exp3.22/global_step51000/converted \
        --vae-dir /llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/ \
        --image-tokenizer-dir /llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/ \
        --input-dir /mmu_mllm_hdd_2/zangdunju/muse/mtp/infer/muse/0.8.1/23000step/greedy/GenEval \
        --output-dir /mmu_mllm_hdd_2/zangdunju/code/dev/muse_v2/exp3.22/0.8.1/23000step/dit20000_1024/greedy/GenEval --image-size 1024 --num-generation-images 1 --max-condition-length 1225

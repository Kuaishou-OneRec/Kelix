PYTHONPATH=. nohup python3 examples/sana/convert_hf_checkpoint.py \
--hf-path /mmu_mllm_hdd_2/yangyiping/models/SANA1.5_4.8B_1024px_diffusers/ \
--output-dir /mmu_mllm_hdd_2/yangyiping/models/SANA1.5_4.8B_1024px_diffusers_muse_converted/ \
> examples/sana/convert_hf_checkpoint.out 2>&1 &
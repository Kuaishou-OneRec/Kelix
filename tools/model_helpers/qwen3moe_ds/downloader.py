# https://huggingface.co/Qwen/Qwen3-30B-A3B


from huggingface_hub import snapshot_download

model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
save_dir = f"/mmu_mllm_hdd_2/lingzhixin/models/DeepSeek-R1-0528-Qwen3-8B"

# deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# 下载整个仓库到指定目录
snapshot_download(
    repo_id=model_name,
    local_dir=save_dir,
    local_dir_use_symlinks=False,  # 在Windows上设为False
    # token="your_huggingface_token"  # 如果需要认证
)

# # 加载下载的模型
# from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(save_dir)
# model = AutoModelForCausalLM.from_pretrained(save_dir)


"""
mkdir -p /mmu_mllm_hdd_2/lingzhixin/models/Keye-30B-A3B-scratch_0612
cd /llm_reco/lingzhixin/pub_models/models; PYTHONPATH=. python3 -m versions.v0_8_1.KeyeQMoe-30B-A3B.tests /mmu_mllm_hdd_2/lingzhixin/models/Keye-30B-A3B-scratch_0612
cd /llm_reco/lingzhixin/recovlm_qw0510/recovlm; PYTHONPATH=. python3 tools/model_helpers/hf_converters/convert_qwen3_to_keye.py --model_dir /mmu_mllm_hdd_2/zhouyang12/models/Qwen3-30B-A3B --vit_model_path /llm_reco_ssd/zangdunju/output2/RecoVLM/SigLIP/siglip_navit/global_step1000/model_float32.pth --new_model_dir /mmu_mllm_hdd_2/lingzhixin/models/Keye-30B-A3B-scratch_0612


cd /llm_reco/lingzhixin/recovlm_qw0510/recovlm; PYTHONPATH=. python3 tools/model_helpers/hf_converters/convert_qwen3_to_keye.py --model_dir /mmu_mllm_hdd_2/lingzhixin/models/DeepSeek-R1-0528-Qwen3-8B --vit_model_path /mmu_mllm_hdd_2/zhouyang12/output/Keye/Stage3_data_0.3.2/0.8.0/206/2b/step6500/converted_hf --new_model_dir /mmu_mllm_hdd_2/lingzhixin/models/Keye-R1-0528-8B-vit0.8.1_0606/


"""
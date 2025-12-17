from recovlm.models.tokenizer_end2end.configuration_keye import KeyeConfig
from recovlm.models.tokenizer_end2end.modeling_keye import KeyeForConditionalGeneration,KeyeImageTokenizer
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig
import random
import torch
from torch import nn


# --- 第 1 步: 加载两个源配置 ---

# 加载 Qwen3 0.6B 的 LLM 配置
llm_model_name = '/llm_reco_ssd/zhouyang12/models/Qwen3-0.6B'
llm_config = AutoConfig.from_pretrained(llm_model_name)

# 加载 Keye 8B 的完整多模态配置
keye_8b_config_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_end2end_init_Keye1_5_init'
# 为了方便演示，我将变量名改为 keye_config_to_modify
keye_config_to_modify = KeyeConfig.from_pretrained(keye_8b_config_path)

# --- 打印修改前的配置以作对比 ---
print("--- 原始 Keye 8B 配置 (LLM 部分) ---")
print(f"Hidden Size: {keye_config_to_modify.hidden_size}")
print(f"Num Hidden Layers: {keye_config_to_modify.num_hidden_layers}")
print(f"Intermediate Size: {keye_config_to_modify.intermediate_size}")

print("\n--- 目标 Qwen3 0.6B 配置 ---")
print(f"Hidden Size: {llm_config.hidden_size}")
print(f"Num Hidden Layers: {llm_config.num_hidden_layers}")
print(f"Intermediate Size: {llm_config.intermediate_size}")

# --- 第 2 步: 将 LLM 配置转换为字典并更新 Keye 配置 ---
llm_config_dict = llm_config.to_dict()

# 使用 update 方法，用 llm_config_dict 中的值覆盖 keye_config_to_modify 中对应的属性
keye_config_to_modify.update(llm_config_dict)

# --- 第 3 步: 验证修改后的配置 ---
print("\n--- 修改后的最终 Keye 配置 (LLM 部分) ---")
print(f"Hidden Size: {keye_config_to_modify.hidden_size}")
print(f"Num Hidden Layers: {keye_config_to_modify.num_hidden_layers}")
print(f"Intermediate Size: {keye_config_to_modify.intermediate_size}")

# 确认 LLM 参数已被更新
assert keye_config_to_modify.hidden_size == llm_config.hidden_size
assert keye_config_to_modify.num_hidden_layers == llm_config.num_hidden_layers

# 确认视觉部分配置仍然存在且未被修改
# (我们可以检查一个视觉部分的特有参数，比如 codebook_size)
print("\n--- 验证视觉部分配置 ---")
original_vision_codebook_size = KeyeConfig.from_pretrained(keye_8b_config_path).vision_config.codebook_size
print(f"视觉部分的 Codebook Size: {keye_config_to_modify.vision_config.codebook_size} (原始尺寸: {original_vision_codebook_size})")
assert keye_config_to_modify.vision_config is not None
assert keye_config_to_modify.vision_config.codebook_size == original_vision_codebook_size

# --- 第 4 步: 保存最终的配置文件 ---
save_directory = "/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_exp_base"
keye_config_to_modify.save_pretrained(save_directory)

print(f"\n成功！新的混合配置文件已保存至: {save_directory}/config.json")
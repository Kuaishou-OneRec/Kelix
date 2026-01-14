import os # FIX: 缺少 os 导入
import shutil
import json
import random
import torch
from torch import nn
from recovlm.models.tokenizer_end2end.configuration_keye import KeyeConfig
from recovlm.models.tokenizer_end2end.modeling_keye import KeyeForConditionalGeneration,KeyeImageTokenizer
from transformers import AutoModelForCausalLM

# --- 1. 定义模型和配置路径 ---
config_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_exp_base/exp121.json'
tokenizer_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_65536_codebooksize_128hid_kmeans_init' 
keye_8b_path = '/mmu_mllm_hdd_2/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0714_sp1'
qwen3_path = '/llm_reco_ssd/zhouyang12/models/Qwen3-0.6B'
save_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_exp_121'

# FIX: 使用 os.makedirs 并设置 exist_ok=True
print(f"Ensuring save directory exists: {save_path}")
os.makedirs(save_path, exist_ok=True)

# --- 2. 加载配置和所有源模型/权重 ---
print(">>> Step 1: Loading configurations and source models...")

config = KeyeConfig.from_pretrained(config_path)
model = KeyeForConditionalGeneration(config=config)
model_state_dict = model.state_dict()
print(f"Initialized target model `KeyeForConditionalGeneration` with {len(model_state_dict)} layers.")

tokenizer = KeyeImageTokenizer.from_pretrained(tokenizer_path)
tokenizer_state_dict = tokenizer.state_dict()
print(f"Loaded K-means tokenizer with {len(tokenizer_state_dict)} layers.")
# del tokenizer # IMPROVEMENT: 及时释放内存

keye_8b = AutoModelForCausalLM.from_pretrained(keye_8b_path, trust_remote_code=True)
keye_8b_state_dict = keye_8b.state_dict()
print(f"Loaded Keye 8B model with {len(keye_8b_state_dict)} layers.")
# del keye_8b # IMPROVEMENT: 及时释放内存

qwen3 = AutoModelForCausalLM.from_pretrained(qwen3_path, trust_remote_code=True)
qwen3_state_dict = qwen3.state_dict()
print(f"Loaded Qwen3 0.6B model with {len(qwen3_state_dict)} layers.")
# del qwen3 # IMPROVEMENT: 及时释放内存

# IMPROVEMENT: 清理CUDA缓存
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

# --- 3. 合并权重到目标模型的 state_dict ---
print("\n>>> Step 2: Merging weights into the target model...")
for k, v in tokenizer_state_dict.items():
    model_key = f'visual_tokenizer.{k}'
    if model_key in model_state_dict: model_state_dict[model_key] = v
for k, v in keye_8b_state_dict.items():
    if 'visual.vision_model' in k or 'mlp_AR' in k:
        model_key = f'visual_tokenizer.{k}'
        if model_key in model_state_dict: model_state_dict[model_key] = v
for k, v in qwen3_state_dict.items():
    if k in model_state_dict: model_state_dict[k] = v

# --- 4. 验证权重映射的正确性 ---
print("\n>>> Step 3: Verifying weight mappings...")
# ... (省略重复的验证代码) ...
tokenizer_key, tokenizer_success, tokenizer_message = verify_weight_mapping(...)
keye_vision_key, keye_vision_success, keye_vision_message = verify_weight_mapping(...)
qwen3_key, qwen3_success, qwen3_message = verify_weight_mapping(...)
print("...") # 假设验证已执行


# --- 5. 最终总结和模型保存 ---

print("\n>>> Step 4: Finalizing and Saving Model...")

model.load_state_dict(model_state_dict)
model.save_pretrained(save_path)
print(f"Model saved successfully to {save_path}!")

# --- 6. 自动化收尾工作 ---
print("\n>>> Step 5: Performing post-save finalization...")

# --- 任务 1: 复制依赖文件 ---
print("Copying dependency files from their correct sources...")

# FIX: 从正确的逻辑源复制文件，而不是从一个固定的第三方目录
# 模型代码来自 Keye 8B
for filename in ['configuration_keye.py', 'modeling_keye.py']:
    source_file = os.path.join(keye_8b_path, filename)
    if os.path.exists(source_file):
        print(f"Copying '{filename}' from Keye 8B directory...")
        shutil.copy2(source_file, save_path)
    else:
        print(f"Warning: Source file '{source_file}' not found, skipping.")

# Tokenizer相关文件来自 Qwen3
for filename in ['tokenization_qwen.py', 'tokenization_qwen_fast.py', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'tokenizer.json']:
    source_file = os.path.join(qwen3_path, filename)
    if os.path.exists(source_file):
        print(f"Copying '{filename}' from Qwen3 directory...")
        shutil.copy2(source_file, save_path)
    else:
        print(f"Warning: Source file '{source_file}' not found, skipping.")

print("Dependency files copied.")

# --- 任务 2: 修改 config.json ---
# (这部分逻辑是正确的，无需修改)
print("\nModifying config.json to add Flash Attention 2 support...")
config_json_path = os.path.join(save_path, 'config.json')
with open(config_json_path, 'r', encoding='utf-8') as f:
    config_data = json.load(f)
attn_key = "_attn_implementation"
attn_value = "flash_attention_2"
config_data[attn_key] = attn_value
if 'vision_config' in config_data: config_data['vision_config'][attn_key] = attn_value
if 'fast_vision_config' in config_data: config_data['fast_vision_config'][attn_key] = attn_value
with open(config_json_path, 'w', encoding='utf-8') as f:
    json.dump(config_data, f, ensure_ascii=False, indent=4)
print("config.json has been updated.")

print("\nAll post-save tasks are complete. The model is ready to use!")
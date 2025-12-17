import os
import shutil
import json
import random
import torch
from torch import nn

# 导入你的自定义模型和配置类
from recovlm.models.tokenizer_end2end.configuration_keye import KeyeConfig
from recovlm.models.tokenizer_end2end.modeling_keye import KeyeForConditionalGeneration,KeyeImageTokenizer

# 导入 Transformers 库
from transformers import AutoModelForCausalLM

# ==============================================================================
# --- 1. 定义所有输入和输出路径 ---
# ==============================================================================
# 最终模型的配置文件
config_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_exp_base/exp121.json'

# 视觉码表 (K-means)
tokenizer_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_65536_codebooksize_128hid_kmeans_init' 

# 视觉编码器 (NaViT) 和 投影层 (mlp_ar)
keye_8b_path = '/mmu_mllm_hdd_2/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0714_sp1'

# 语言模型 (LLM)
qwen3_path = '/llm_reco_ssd/zhouyang12/models/Qwen3-0.6B'

# 最终合并后模型的保存路径
save_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_exp_121'

dependency_source_path = '/llm_reco_ssd/zhouyang12/models/KeyeImageTokenizer_end2end_init_Keye1_5_base_init'


# ==============================================================================
# --- 2. 准备工作：创建保存目录 ---
# ==============================================================================
print(f"Ensuring save directory exists: {save_path}")
# 使用 os.makedirs 并设置 exist_ok=True，可以安全地创建目录，如果目录已存在也不会报错
os.makedirs(save_path, exist_ok=True)


# ==============================================================================
# --- 3. 加载配置和所有源模型/权重 ---
# ==============================================================================
print("\n>>> Step 1: Loading configurations and source models...")

# 加载目标模型的配置，并创建一个空的模型
config = KeyeConfig.from_pretrained(config_path)
model = KeyeForConditionalGeneration(config=config)
model_state_dict = model.state_dict()
print(f"Initialized target model `KeyeForConditionalGeneration` with {len(model_state_dict)} layers.")


# 加载视觉码表 (K-means)
tokenizer = KeyeImageTokenizer.from_pretrained(tokenizer_path)
tokenizer_state_dict = tokenizer.state_dict()
print(f"Loaded K-means tokenizer with {len(tokenizer_state_dict)} layers.")

# 加载包含 NaViT 和 mlp_ar 的 Keye 8B 模型
keye_8b = AutoModelForCausalLM.from_pretrained(keye_8b_path, trust_remote_code=True)
keye_8b_state_dict = keye_8b.state_dict()
print(f"Loaded Keye 8B model with {len(keye_8b_state_dict)} layers.")

# 加载 Qwen3 0.6B LLM
qwen3 = AutoModelForCausalLM.from_pretrained(qwen3_path, trust_remote_code=True)
qwen3_state_dict = qwen3.state_dict()
print(f"Loaded Qwen3 0.6B model with {len(qwen3_state_dict)} layers.")




# ==============================================================================
# --- 4. 合并权重到目标模型的 state_dict ---
# ==============================================================================
print("\n>>> Step 2: Merging weights into the target model...")

# Part A: 加载视觉码表 (K-means)
print("Merging K-means codebook weights...")
for k, v in tokenizer_state_dict.items():
    model_key = f'visual_tokenizer.{k}'
    if model_key in model_state_dict:
        model_state_dict[model_key] = v

# Part B: 加载视觉编码器 (NaViT) 和 投影层 (mlp_ar)
print("Merging NaViT and projector weights from Keye 8B...")
for k, v in keye_8b_state_dict.items():
    if 'visual.vision_model' in k or 'mlp_AR' in k:
        model_key = f'visual_tokenizer.{k}'
        if model_key in model_state_dict:
            model_state_dict[model_key] = v

# Part C: 加载 Qwen3 LLM 的权重 (这将覆盖所有LLM相关的层)
print("Merging LLM weights from Qwen3 0.6B...")
for k, v in qwen3_state_dict.items():
    if k in model_state_dict:
        model_state_dict[k] = v


# ==============================================================================
# --- 5. 验证权重映射的正确性 ---
# ==============================================================================
def verify_weight_mapping(source_dict, target_dict, key_mapping_func=None, source_keys_filter=None):
    if not source_dict: return None, False, "Source dictionary is empty"
    valid_keys = list(source_dict.keys())
    if source_keys_filter: valid_keys = [k for k in valid_keys if source_keys_filter(k)]
    if not valid_keys: return None, False, "No valid keys found in source dictionary after filtering"
    random_key = random.choice(valid_keys)
    source_tensor = source_dict[random_key]
    target_key = key_mapping_func(random_key) if key_mapping_func else random_key
    if target_key not in target_dict: return random_key, False, f"Key '{target_key}' not found in target dictionary"
    target_tensor = target_dict[target_key]
    if torch.allclose(source_tensor, target_tensor): return random_key, True, f"Success: '{target_key}' matches '{random_key}'"
    else: return random_key, False, f"Mismatch: '{target_key}' does not match '{random_key}'"

print("\n>>> Step 3: Verifying weight mappings...")

tokenizer_key, tokenizer_success, tokenizer_message = verify_weight_mapping(tokenizer_state_dict, model_state_dict, key_mapping_func=lambda k: f"visual_tokenizer.{k}", source_keys_filter=lambda k: 'quantizer.embedding' in k)
print(f"Verifying K-means Codebook: {'✓ Success' if tokenizer_success else '✗ Failed'}. (Layer: {tokenizer_key}, Msg: {tokenizer_message})")

keye_vision_key, keye_vision_success, keye_vision_message = verify_weight_mapping(keye_8b_state_dict, model_state_dict, source_keys_filter=lambda k: 'visual.vision_model' in k or 'mlp_AR' in k, key_mapping_func=lambda k: f'visual_tokenizer.{k}')
print(f"Verifying Keye Vision/Projector: {'✓ Success' if keye_vision_success else '✗ Failed'}. (Layer: {keye_vision_key}, Msg: {keye_vision_message})")

qwen3_key, qwen3_success, qwen3_message = verify_weight_mapping(qwen3_state_dict, model_state_dict, key_mapping_func=lambda k: k, source_keys_filter=lambda k: not k.startswith('visual'))
print(f"Verifying Qwen3 LLM: {'✓ Success' if qwen3_success else '✗ Failed'}. (Layer: {qwen3_key}, Msg: {qwen3_message})")


# ==============================================================================
# --- 6. 最终总结和模型保存 ---
# ==============================================================================
print("\n>>> Step 4: Finalizing and Saving Model...")

print("\n=== Verification Summary ===")
if tokenizer_success and keye_vision_success and qwen3_success:
    print("✓ All components were correctly mapped into the final model's state_dict.")
else:
    print("✗ One or more components were not correctly mapped. Please check the logs.")

print("Initializing `quant_projector` weights...")
if 'quant_projector.weight' in model_state_dict:
    nn.init.kaiming_normal_(model_state_dict['quant_projector.weight'], a=0, mode='fan_in', nonlinearity='relu')
    print("`quant_projector.weight` initialized.")
else:
    print("Warning: `quant_projector.weight` not found in model state dict.")

print("\nLoading final state_dict and saving model...")
model.load_state_dict(model_state_dict)
model.save_pretrained(save_path)
print(f"Model saved successfully to '{save_path}'!")


# ==============================================================================
# --- 7. 自动化收尾工作：复制依赖文件 & 修改配置 ---
# ==============================================================================
print("\n>>> Step 5: Performing post-save finalization...")

# --- 任务 1: 从单一模板目录复制所有依赖文件 ---
print(f"\nCopying all dependency files from the source directory: '{dependency_source_path}'...")

files_to_skip = ['config.json', 'model.safetensors.index.json',]

if not os.path.isdir(dependency_source_path):
    print(f"\nError: Dependency source directory not found at '{dependency_source_path}'. SKIPPING FILE COPY.\n")
else:
    for item_name in os.listdir(dependency_source_path):
        if item_name in files_to_skip or 'safetensors' in item_name:
            print(f"  - Skipping '{item_name}' (generated by script).")
            continue
        
        source_item = os.path.join(dependency_source_path, item_name)
        target_item = os.path.join(save_path, item_name)

        if os.path.isdir(source_item):
            print(f"  - Copying directory: '{item_name}'")
            shutil.copytree(source_item, target_item, dirs_exist_ok=True)
        else:
            print(f"  - Copying file: '{item_name}'")
            shutil.copy2(source_item, target_item)
    print("Dependency files copied.")

# --- 任务 2: 修改 config.json 以添加 flash_attention_2 支持 ---
print("\nModifying config.json to add Flash Attention 2 support...")
config_json_path = os.path.join(save_path, 'config.json')

with open(config_json_path, 'r', encoding='utf-8') as f:
    config_data = json.load(f)

attn_key = "_attn_implementation"
attn_value = "flash_attention_2"
config_data[attn_key] = attn_value
if 'vision_config' in config_data and isinstance(config_data['vision_config'], dict):
    config_data['vision_config'][attn_key] = attn_value
if 'fast_vision_config' in config_data and isinstance(config_data['fast_vision_config'], dict):
    config_data['fast_vision_config'][attn_key] = attn_value

with open(config_json_path, 'w', encoding='utf-8') as f:
    json.dump(config_data, f, ensure_ascii=False, indent=4)

print("config.json has been updated with Flash Attention 2 settings.")

print("\n\n==============================================================================")
print("             All tasks are complete. The model is ready to use!             ")
print(f"             Final model located at: {save_path}             ")
print("==============================================================================")
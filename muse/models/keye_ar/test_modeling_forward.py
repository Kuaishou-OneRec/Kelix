"""
KeyeARModel前向demo脚本
基于test_ar_ori_forward.py修改，适配modeling.py中的KeyeARModel
"""

import os
import json
import torch
from transformers import AutoProcessor
from muse.models.keye_ar.modeling import KeyeARModel
from muse.config import UnifiedQwen3Config
from keye_vl_utils import process_vision_info

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'

def get_config_value(conf_data, key, default_value, new_key=None, source_key=None):
    """从配置数据中获取值，如果不存在则使用默认值并打印警告"""
    if key in conf_data:
        return conf_data[key]
    else:
        source = source_key if source_key else key
        target = new_key if new_key else key
        print(f"Warning: Using default value for {target} (from {source}): {default_value}")
        return default_value

def convert_conf_to_unified_qwen3_config(conf_path):
    """将conf.json转换为适合UnifiedQwen3Config的配置字段"""
    with open(conf_path, 'r') as f:
        conf_data = json.load(f)
    
    # 提取顶层配置字段
    config_dict = {}
    
    # 基础模型配置
    config_dict['vocab_size'] = get_config_value(conf_data, 'vocab_size', 151936)
    config_dict['embed_dim'] = get_config_value(conf_data, 'hidden_size', 4096, 'embed_dim', 'hidden_size')
    config_dict['num_layers'] = get_config_value(conf_data, 'num_hidden_layers', 36, 'num_layers', 'num_hidden_layers')
    config_dict['num_heads'] = get_config_value(conf_data, 'num_attention_heads', 32, 'num_heads', 'num_attention_heads')
    config_dict['num_kv_heads'] = get_config_value(conf_data, 'num_key_value_heads', 8, 'num_kv_heads', 'num_key_value_heads')
    config_dict['head_dim'] = get_config_value(conf_data, 'head_dim', 128)
    config_dict['intermediate_dim'] = get_config_value(conf_data, 'intermediate_size', 12288, 'intermediate_dim', 'intermediate_size')
    config_dict['hidden_act'] = get_config_value(conf_data, 'hidden_act', 'silu')
    config_dict['norm_eps'] = get_config_value(conf_data, 'rms_norm_eps', 1e-6, 'norm_eps', 'rms_norm_eps')
    config_dict['attn_dropout'] = get_config_value(conf_data, 'attention_dropout', 0.0)
    config_dict['attention_function'] = get_config_value(conf_data, 'attention_function', 'eager')
    config_dict['q_proj_bias'] = get_config_value(conf_data, 'q_proj_bias', False)
    config_dict['k_proj_bias'] = get_config_value(conf_data, 'k_proj_bias', False)
    config_dict['v_proj_bias'] = get_config_value(conf_data, 'v_proj_bias', False)
    config_dict['attention_bias'] = get_config_value(conf_data, 'attention_bias', False)
    config_dict['tie_word_embeddings'] = get_config_value(conf_data, 'tie_word_embeddings', True)
    
    # RoPE配置
    config_dict['rope_theta'] = get_config_value(conf_data, 'rope_theta', 1000000)
    config_dict['max_seq_len'] = get_config_value(conf_data, 'max_position_embeddings', 40960, 'max_seq_len', 'max_position_embeddings')
    config_dict['rope_base'] = get_config_value(conf_data, 'rope_theta', 1000000, 'rope_base', 'rope_theta')
    
    # RoPE scaling配置
    if 'rope_scaling' in conf_data:
        config_dict['rope_scaling'] = conf_data['rope_scaling']
        if 'mrope_section' in conf_data['rope_scaling']:
            config_dict['mrope_section'] = conf_data['rope_scaling']['mrope_section']
    
    # 特殊token IDs
    config_dict['image_token_id'] = conf_data.get('image_token_id')
    config_dict['pad_token_id'] = conf_data.get('pad_token_id')
    config_dict['q_eos_token'] = conf_data.get('q_eos_token')
    
    # Vision配置（从vision_config中提取）
    if 'vision_config' in conf_data:
        vision_config = conf_data['vision_config']
        config_dict['codebook_size'] = get_config_value(vision_config, 'codebook_size', 8192)  # 默认值改为8192
        config_dict['n_q_tokens'] = get_config_value(vision_config, 'n_q_tokens', 8)
    else:
        print("Warning: vision_config not found in conf_data, using default values for all vision-related configs")
        config_dict['codebook_size'] = 8192  # 默认值改为8192
        config_dict['n_q_tokens'] = 8
    
    # Token head配置
    config_dict['token_head_d_model'] = get_config_value(conf_data, 'token_head_dim', 512, 'token_head_d_model', 'token_head_dim')
    config_dict['token_head_nheads'] = get_config_value(conf_data, 'token_head_nhead', 4, 'token_head_nheads', 'token_head_nhead')
    config_dict['token_head_dim_feedforward'] = get_config_value(conf_data, 'token_head_intermediate_dim', 2048, 'token_head_dim_feedforward', 'token_head_intermediate_dim')
    config_dict['token_head_num_layers'] = get_config_value(conf_data, 'token_head_num_layers', 3)
    config_dict['token_head_attention_function'] = get_config_value(conf_data, 'token_head_attention_function', 'eager')
    config_dict['token_head_use_gradient_checkpointing'] = get_config_value(conf_data, 'token_head_use_gradient_checkpointing', True)
    config_dict['token_head_reduce'] = get_config_value(conf_data, 'token_head_reduce', True)
    
    # 其他配置
    config_dict['use_multimodal_rope'] = True  # 根据Qwen3Config默认值设置
    
    return config_dict

def compare_config_fields():
    """比较UnifiedQwen3Config字段与转换后配置字段的差异"""
    # 获取UnifiedQwen3Config的所有字段
    unified_config_fields = set(UnifiedQwen3Config.model_fields.keys())
    
    # 获取转换后的配置字段
    converted_config = convert_conf_to_unified_qwen3_config("muse/models/keye_ar/conf.json")
    converted_fields = set(converted_config.keys())
    
    # 找出缺失的字段
    missing_fields = unified_config_fields - converted_fields
    extra_fields = converted_fields - unified_config_fields
    
    print("=" * 50)
    print("配置字段对比结果:")
    print("=" * 50)
    print(f"UnifiedQwen3Config总字段数: {len(unified_config_fields)}")
    print(f"转换后配置字段数: {len(converted_fields)}")
    print()
    
    if missing_fields:
        print("缺失的字段 (在UnifiedQwen3Config中但不在转换后配置中):")
        for field in sorted(missing_fields):
            print(f"  - {field}")
        print()
    
    if extra_fields:
        print("多余的字段 (在转换后配置中但不在UnifiedQwen3Config中):")
        for field in sorted(extra_fields):
            print(f"  - {field}")
        print()
    
    if not missing_fields and not extra_fields:
        print("字段完全匹配!")
    
    print("统一的字段:")
    common_fields = unified_config_fields & converted_fields
    for field in sorted(common_fields):
        print(f"  - {field}")
    
    return {
        "unified_config_fields": unified_config_fields,
        "converted_fields": converted_fields,
        "missing_fields": missing_fields,
        "extra_fields": extra_fields
    }

def load_keye_ar_model():
    """加载KeyeARModel，使用convert_hf_state_dict函数转换权重"""
    # 使用与test_ar_ori_forward.py相同的模型路径
    output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"
    
    # 加载processor和配置
    processor = AutoProcessor.from_pretrained(output_model_dir, trust_remote_code=True)

    # 使用新的转换函数加载配置
    config_dict = convert_conf_to_unified_qwen3_config("muse/models/keye_ar/conf.json")
    config = UnifiedQwen3Config(**config_dict)
    
    # 创建模型实例
    model = KeyeARModel(config)
    
    # 加载并转换状态字典
    state_dict = torch.load(f"{output_model_dir}/pytorch_model.bin", map_location="cpu")
    converted_state_dict = KeyeARModel.convert_hf_state_dict(state_dict, tie_word_embeddings=False)
    model.load_state_dict(converted_state_dict, strict=False)
    
    # 将模型移到设备并转换为bfloat16精度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).bfloat16()
    
    return model, processor, device

def process_message(messages, processor, device, add_generation_prompt=True, padding=False):
    """处理消息，生成模型输入"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=padding,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    
    # 转换到bfloat16精度
    def _cast_inputs_to_bf16(batch):
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.to(dtype=torch.bfloat16)
        return batch
    
    inputs = _cast_inputs_to_bf16(inputs)
    # 确保 inputs 全部在目标 device 上
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

def test_forward():
    """测试KeyeARModel前向传播"""
    model, processor, device = load_keye_ar_model()
    
    # 使用与test_ar_ori_forward.py相同的消息格式
    COT_SYSTEM_PROMPT = "You are a helpful assistant."
    messages = [
        {"role": "system",
         "content": [
             {"type": "text", "text": COT_SYSTEM_PROMPT},
         ], },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": " What's sum of the first 10 positive integers? After necessary analysis, your final output should follow the format: Final Answer: X."},
            ],
        }
    ]
    
    inputs = process_message(messages, processor, device)
    
    # 使用与test_ar_ori_forward.py相同的autocast逻辑
    if torch.cuda.is_available():
        autocast_cm = torch.cuda.amp.autocast
    else:
        # CPU 上也可以使用 bfloat16 autocast（需要对应 PyTorch 版本）
        try:
            autocast_cm = torch.cpu.amp.autocast
        except Exception:
            from contextlib import nullcontext
            autocast_cm = nullcontext  # fallback，若没有 cpu autocast 则不使用

    with autocast_cm(dtype=torch.bfloat16):
        outputs = model(**inputs)
    
    print(f"Output type: {type(outputs)}")
    # KeyeARModel的forward方法返回的是Qwen3Model的输出，其中包含hidden_states
    if hasattr(outputs, 'last_hidden_state'):
        print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")

if __name__ == "__main__":
    # 运行字段对比
    # compare_config_fields()
    
    # 如果需要测试前向传播，取消下面的注释
    test_forward()
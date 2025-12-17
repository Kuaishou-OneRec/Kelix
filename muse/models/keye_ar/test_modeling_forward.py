"""
KeyeARModel前向demo脚本
基于test_ar_ori_forward.py修改，适配modeling.py中的KeyeARModel
"""

import os
import json
import torch
from transformers import AutoProcessor
from muse.models.keye_ar.modeling import KeyeARModel
from muse.config import KeyeARConfig, UnifiedQwen3Config, KeyeTokenizerConfig, UnifiedTokenDecoderConfig, KeyeVisionConfig

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'

def load_keye_ar_config(conf_path):
    """直接从conf.json加载KeyeARConfig"""
    with open(conf_path, 'r') as f:
        conf_data = json.load(f)
    
    # 构造KeyeVisionConfig
    vision_config_data = conf_data.get('vision_config', {})
    vision_config_inner = vision_config_data.get('vision_config', {})
    keye_vision_config = KeyeVisionConfig(
        image_size=vision_config_inner.get('image_size', 384),
        patch_size=vision_config_inner.get('patch_size', 14),
        hidden_size=vision_config_inner.get('hidden_size', 1152),
        num_hidden_layers=vision_config_inner.get('num_hidden_layers', 27),
        num_attention_heads=vision_config_inner.get('num_attention_heads', 16),
        intermediate_size=vision_config_inner.get('intermediate_size', 4304),
        hidden_act=vision_config_inner.get('hidden_act', 'gelu_pytorch_tanh'),
        layer_norm_eps=vision_config_inner.get('layer_norm_eps', 1e-6),
        attention_dropout=vision_config_inner.get('attention_dropout', 0.0),
        rope_theta=vision_config_inner.get('rope_theta', 10000.0),
    )
    
    # 构造KeyeTokenizerConfig
    keye_tokenizer_config = KeyeTokenizerConfig(
        vision_config=keye_vision_config,
        codebook_size=vision_config_data.get('codebook_size', 65536),
        embedding_dim=vision_config_data.get('embedding_dim', 128),
        init_embedding_dim=vision_config_data.get('init_embedding_dim', 4096),
        llm_hidden_size=vision_config_data.get('llm_hidden_size', 4096),
        n_q_tokens=vision_config_data.get('n_q_tokens', 8),
    )
    
    # 构造UnifiedTokenDecoderConfig
    unified_token_decoder_config = UnifiedTokenDecoderConfig(
        vocab_size=vision_config_data.get('codebook_size', 65536),
        d_model=conf_data.get('token_head_dim', 512),
        nhead=conf_data.get('token_head_nhead', 4),
        num_layers=3,  # 默认值
        dim_feedforward=conf_data.get('token_head_intermediate_dim', 2048),
    )
    
    # 构造UnifiedQwen3Config
    unified_qwen_config = UnifiedQwen3Config(
        vocab_size=conf_data.get('vocab_size', 151936),
        embed_dim=conf_data.get('hidden_size', 4096),
        num_layers=conf_data.get('num_hidden_layers', 36),
        num_heads=conf_data.get('num_attention_heads', 32),
        num_kv_heads=conf_data.get('num_key_value_heads', 8),
        head_dim=conf_data.get('head_dim', 128),
        intermediate_dim=conf_data.get('intermediate_size', 12288),
        hidden_act=conf_data.get('hidden_act', 'silu'),
        norm_eps=conf_data.get('rms_norm_eps', 1e-6),
        attn_dropout=conf_data.get('attention_dropout', 0.0),
        rope_theta=conf_data.get('rope_theta', 1000000),
        max_seq_len=conf_data.get('max_position_embeddings', 40960),
        image_token_id=conf_data.get('image_token_id'),
        pad_token_id=conf_data.get('pad_token_id'),
        q_eos_token=conf_data.get('q_eos_token'),
        codebook_size=vision_config_data.get('codebook_size', 65536),
        n_q_tokens=vision_config_data.get('n_q_tokens', 8),
        token_head_d_model=conf_data.get('token_head_dim', 512),
        token_head_nheads=conf_data.get('token_head_nhead', 4),
        token_head_dim_feedforward=conf_data.get('token_head_intermediate_dim', 2048),
        token_head_num_layers=conf_data.get('token_head_num_layers', 3),
    )
    
    # 构造KeyeARConfig
    keye_ar_config = KeyeARConfig(
        qwen_config=unified_qwen_config,
        tokenizer_config=keye_tokenizer_config,
        token_decoder_config=unified_token_decoder_config,
    )
    
    return keye_ar_config

def load_keye_ar_model():
    """加载KeyeARModel，使用convert_hf_state_dict函数转换权重"""
    # 使用与test_ar_ori_forward.py相同的模型路径
    output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"
    
    # 加载processor和配置
    processor = AutoProcessor.from_pretrained(output_model_dir, trust_remote_code=True)

    # 直接从conf.json加载KeyeARConfig
    config = load_keye_ar_config("muse/models/keye_ar/conf.json")
    
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
    # 如果需要测试前向传播，取消下面的注释
    test_forward()
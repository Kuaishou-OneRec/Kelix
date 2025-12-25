"""
Qwen3模型前向计算的demo，基于test_qwen3.py的参数加载逻辑
"""

import os
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from muse.config import Qwen3Config
from muse.models.qwen3 import Qwen3Model
from muse.training.common import set_default_dtype


def _build_qwen3_config(hf_cfg: Dict[str, Any]) -> Qwen3Config:
    """Map Hugging Face config to Muse Qwen3Config."""
    embed_dim = hf_cfg.get("hidden_size") or hf_cfg.get("dim")
    num_heads = hf_cfg.get("num_attention_heads") or hf_cfg.get("n_head")
    num_layers = hf_cfg.get("num_hidden_layers") or hf_cfg.get("n_layer")
    num_kv_heads = (
        hf_cfg.get("num_key_value_heads")
        or hf_cfg.get("n_kv_head")
        or num_heads
    )
    head_dim = hf_cfg.get("head_dim") or (embed_dim // num_heads)
    intermediate_dim = hf_cfg.get("intermediate_size") or hf_cfg.get(
        "ffn_hidden_size", 4 * embed_dim
    )
    max_seq_len = (
        hf_cfg.get("max_position_embeddings")
        or hf_cfg.get("max_seq_len")
        or 32768
    )
    rope_base = hf_cfg.get("rope_theta", hf_cfg.get("rotary_emb_base", 10000.0))
    attn_dropout = hf_cfg.get(
        "attention_dropout",
        hf_cfg.get("attention_dropout_prob", 0.0),
    )
    qkv_bias = hf_cfg.get("use_qkv_bias")
    q_norm_flag = hf_cfg.get("use_qk_norm", hf_cfg.get("qk_norm", True))

    attention_function = (
        "flash_attention_2" if hf_cfg.get("use_flash_attn", False) else "eager"
    )

    return Qwen3Config(
        model_class="Qwen3Model",
        vocab_size=hf_cfg["vocab_size"],
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_dim,
        max_seq_len=max_seq_len,
        rope_base=rope_base,
        norm_eps=hf_cfg.get("rms_norm_eps", 1e-6),
        attn_dropout=attn_dropout,
        tie_word_embeddings=hf_cfg.get("tie_word_embeddings", True),
        q_proj_bias=hf_cfg.get("q_proj_bias", qkv_bias or False),
        k_proj_bias=hf_cfg.get("k_proj_bias", qkv_bias or False),
        v_proj_bias=hf_cfg.get("v_proj_bias", qkv_bias or False),
        attention_function=attention_function,
        q_norm=q_norm_flag,
        k_norm=q_norm_flag,
    )


def demo_qwen3_forward():
    """
    Qwen3模型前向计算的demo
    """
    print("=" * 60)
    print("Qwen3模型前向计算Demo")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    # 检查预训练模型路径是否存在
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base"
    if not os.path.exists(checkpoint_dir):
        print(f"错误：预训练模型路径不存在: {checkpoint_dir}")
        print("请修改checkpoint_dir为正确的预训练模型路径")
        return
    
    print(f"加载预训练模型: {checkpoint_dir}")
    
    # 加载Hugging Face模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    # 获取设备和数据类型
    device = next(hf_model.parameters()).device
    dtype = next(hf_model.parameters()).dtype
    print(f"模型设备: {device}, 数据类型: {dtype}")
    
    # 准备输入文本
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用chat template并编码
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    print(f"输入文本: {text}")
    print(f"输入token数量: {model_inputs['input_ids'].shape[1]}")
    
    # 加载HF模型的配置和state dict
    hf_state_dict = hf_model.state_dict()
    hf_config_dict = hf_model.config.to_dict()
    
    # 构建Muse模型配置
    muse_config = _build_qwen3_config(hf_config_dict)
    print(f"Muse模型配置构建完成，层数: {muse_config.num_layers}")
    
    # 创建Muse模型实例
    model_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    with set_default_dtype(model_dtype):
        muse_model = Qwen3Model(muse_config)
    
    # 转换并加载state dict
    print("转换并加载预训练权重...")
    converted_state_dict = muse_model.convert_hf_state_dict(hf_state_dict)
    
    # 将权重加载到Muse模型
    missing_keys, unexpected_keys = muse_model.load_state_dict(
        converted_state_dict, strict=False
    )
    
    # 检查权重加载情况
    print(f"权重加载结果:")
    print(f"  缺失的键: {len(missing_keys)}")
    print(f"  意外的键: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"  部分缺失键示例: {missing_keys[:5]}")
    if unexpected_keys:
        print(f"  部分意外键示例: {unexpected_keys[:5]}")
    
    # 将模型移动到设备
    muse_model = muse_model.to(device).to(dtype)
    print(f"Muse模型已移动到设备: {device}，数据类型: {dtype}")
    
    # 进行前向计算
    print("\n进行模型前向计算...")
    
    # Muse模型的输入格式调整
    muse_inputs = model_inputs.copy()
    if "token_type_ids" in muse_inputs:
        del muse_inputs["token_type_ids"]
    
    # 添加position_ids（如果需要）
    if "position_ids" not in muse_inputs:
        seq_length = muse_inputs["input_ids"].shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
        muse_inputs["position_ids"] = position_ids
    
    # 设置为评估模式
    hf_model.eval()
    muse_model.eval()
    
    # HF模型前向计算
    with torch.no_grad():
        hf_outputs = hf_model(**model_inputs)
        hf_logits = hf_outputs.logits
    
    # Muse模型前向计算
    with torch.no_grad():
        muse_outputs = muse_model(**muse_inputs)
        muse_logits = muse_outputs
    
    # 比较logits
    print("\nLogits比较结果:")
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"Muse logits shape: {muse_logits.shape}")
    
    # 计算差异
    if hf_logits.shape == muse_logits.shape:
        logits_diff = (hf_logits - muse_logits).abs()
        print(f"Logits最大差异: {logits_diff.max().item():.6e}")
        print(f"Logits平均差异: {logits_diff.mean().item():.6e}")
        
        # 检查是否接近
        if torch.allclose(hf_logits, muse_logits, atol=1e-3, rtol=1e-3):
            print("✅ Logits匹配度符合要求！")
        else:
            print("⚠️ Logits差异较大")
    
    # 生成示例文本
    print("\n生成示例文本:")
    with torch.no_grad():
        generated_ids = muse_model.generate(
            **muse_inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.95
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"生成结果: {generated_text}")
    
    print("\n" + "=" * 60)
    print("Qwen3前向计算Demo完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_qwen3_forward()
"""
Qwen3模型前向计算的demo，严格参考tests/test_qwen3.py的参数加载逻辑
"""

import os
from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
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

    # 强制使用flash attention 2
    attention_function = "flash_attention_2"

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
    Qwen3模型前向计算的demo，严格参考tests/test_qwen3.py的实现
    """

    # 设置随机种子
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = "cuda:0"
    # 检查预训练模型路径是否存在
    checkpoint_dir = "/llm_reco_ssd/zhouyang12/models/muse/Qwen3-8B-Base"

    # 加载Hugging Face模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    transformers_model = AutoModelForCausalLM.from_pretrained('/llm_reco_ssd/zhouyang12/models/Qwen3-8B-Base').to(device).bfloat16()
    transformers_model.eval()

    
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
    print(f"model_inputs={model_inputs}")
    
    # 创建Muse模型实例
    model_dtype = torch.bfloat16
    with set_default_dtype(model_dtype):
        muse_model = Qwen3Model.from_pretrained(checkpoint_dir).to(device)
        
    # 调用generate函数生成文本
    print("\n" + "=" * 60)
    print("使用Muse模型生成文本")
    print("=" * 60)
    
    # 设置生成参数
    generate_params = {
        "max_new_tokens": 40,
        "temperature": 0.8,
        "top_k": 1,
        "top_p": 0.95,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    print(f"Qwen3 baseline generation:")
    outputs = transformers_model.generate(
            model_inputs["input_ids"],
            max_new_tokens=40,
            do_sample=False,
        )
    print(f"Qwen3 baseline outputs: {outputs}")
    

    # 生成文本
    print(f"生成参数: {generate_params}")
    print("开始生成...")
    
    generated_ids = generate(
        muse_model, 
        model_inputs["input_ids"], 
        **generate_params
    )
    assert torch.all(torch.tensor(generated_ids).to(device) == torch.tensor(outputs).to(device))
    print(f"generated_ids: {generated_ids}")
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n生成结果:")
    print(generated_text)
    
    print("\n" + "=" * 60)
    print("Qwen3模型Demo完成")
    print("=" * 60)



def generate(
    model,  # muse.models.qwen3.Qwen3Model
    input_ids: torch.Tensor,
    max_length: int = 512,
    max_new_tokens: Optional[int] = None,  # 添加max_new_tokens参数
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    **kwargs
) -> List[torch.Tensor]:
    """
    使用Muse模型进行文本生成

    Args:
        model: Muse模型实例 (muse.models.qwen3.Qwen3Model)
        input_ids: 输入token ids，形状为 [batch_size, seq_length]
        max_length: 生成序列的最大长度
        max_new_tokens: 要生成的新token数量
        temperature: 采样温度
        top_k: 仅考虑概率最高的k个token
        top_p: 仅考虑累积概率达到p的token
        eos_token_id: 结束token id
        pad_token_id: 填充token id
        **kwargs: 其他传递给模型的参数

    Returns:
        生成的token序列列表，每个元素形状为 [seq_length]
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    input_seq_len = input_ids.size(1)

    # 处理max_new_tokens参数
    if max_new_tokens is not None:
        # 如果指定了max_new_tokens，计算生成的总长度
        max_length = input_seq_len + max_new_tokens

    # 设置最大生成长度
    if max_length <= input_seq_len:
        return input_ids.tolist()

    # 初始化生成的序列
    generated = input_ids.clone()

    # 设置KV缓存
    model.model.setup_caches(
        batch_size=batch_size,
        dtype=next(model.parameters()).dtype,
        decoder_max_seq_len=max_length
    )

    # 预热阶段 - 处理输入序列，填充KV缓存
    with torch.no_grad():
        # 创建预热阶段的位置id (从0到input_seq_len-1)
        prefill_pos = torch.arange(input_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        # 使用完整的model()调用，并提供正确的input_pos
        model(generated, input_pos=prefill_pos,# is_causal=True
              **kwargs)

    # 自回归生成阶段
    for step in range(input_seq_len, max_length):
        with torch.no_grad():
            # 当前的token是生成序列的最后一个token
            current_token = generated[:, -1:]
            # 计算当前的位置id (应该是step，因为是接着输入序列之后的位置)
            current_pos = torch.tensor([[step]], device=device).expand(batch_size, -1)

            # 前向传播 - 使用完整的model()调用，提供当前位置id
            logits = model(current_token, input_pos=current_pos,#  is_causal=True
                            **kwargs)

            # 采样下一个token
            next_token_logits = logits[:, -1, :]

            # 应用温度缩放
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # 应用top-k采样
            if top_k is not None:
                # 修复：将非top-k的logits设置为负无穷，而不是只保留top-k的值
                # 这样可以保持原始的token索引信息
                top_k_logits, _ = next_token_logits.topk(top_k, dim=-1)
                # 创建掩码：将低于top-k阈值的logits设为负无穷
                next_token_logits = torch.where(
                    next_token_logits >= top_k_logits[:, -1:], 
                    next_token_logits, 
                    -float("inf")
                )
            
            # 应用top-p采样 (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 移除累积概率超过p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 确保至少保留一个token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                # 修复：使用scatter将掩码映射回原始token索引空间
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                indices_to_remove = indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

            # 添加安全检查：确保至少有一个有效token
            if (next_token_logits == -float("inf")).all(): 
                next_token_logits = torch.zeros_like(next_token_logits)

            # 计算概率分布
            probs = F.softmax(next_token_logits, dim=-1)

            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)

            # 将生成的token添加到序列中
            generated = torch.cat([generated, next_token], dim=1)

            # 检查是否所有序列都已生成结束token
            if eos_token_id is not None:
                done = (next_token == eos_token_id).any(dim=1).all()
                if done:
                    break


    model.model.reset_caches()

    # 将生成的序列转换为列表
    generated_list = generated.tolist()

    return generated_list


if __name__ == "__main__":
    demo_qwen3_forward()
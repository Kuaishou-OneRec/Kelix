"""
Qwen3模型前向计算的demo，严格参考tests/test_qwen3.py的参数加载逻辑
"""

import os
from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from muse.config import KeyeARConfig
from muse.models.keye_ar import KeyeARModel
from muse.training.common import set_default_dtype


def demo_keyear_forward():
    """
    KeyeAR模型前向计算的demo，严格参考tests/test_keyear.py的实现
    """

    # 设置随机种子
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = "cuda:0"
    # 检查预训练模型路径是否存在
    checkpoint_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"

    # 加载Hugging Face模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    
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
        muse_model = KeyeARModel.from_pretrained(checkpoint_dir).to(device)
        
    # 调用generate函数生成文本
    print("\n" + "=" * 60)
    print("使用Muse模型生成文本")
    print("=" * 60)
    
    # 设置生成参数
    generate_params = {
        "max_new_tokens": 20,
        "temperature": 0.8,
        "top_k": 1,
        "top_p": 0.95,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    print(f"Qwen3 baseline generation:")

    # 生成文本
    print(f"生成参数: {generate_params}")
    print("开始生成...")
    
    generated_ids = muse_model.generate(
        model_inputs["input_ids"], 
        **generate_params
    )
    print(f"generated_ids: {generated_ids}")
    # assert torch.all(torch.tensor(generated_ids).to(device) == torch.tensor(outputs).to(device))

    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0,...,0], skip_special_tokens=True)
    
    print("\n生成结果:")
    print(generated_text)
    
    print("\n" + "=" * 60)
    print("Qwen3模型Demo完成")
    print("=" * 60)



if __name__ == "__main__":
    demo_keyear_forward()
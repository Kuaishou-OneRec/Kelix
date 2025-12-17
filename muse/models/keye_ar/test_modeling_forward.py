"""
KeyeARModel前向demo脚本
基于test_ar_ori_forward.py修改，适配modeling.py中的KeyeARModel
"""

import os
import torch
from transformers import AutoProcessor
from muse.models.keye_ar.modeling import KeyeARModel
from muse.config import UnifiedQwen3Config
from keye_vl_utils import process_vision_info

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'

def load_keye_ar_model():
    """加载KeyeARModel，使用convert_hf_state_dict函数转换权重"""
    # 使用与test_ar_ori_forward.py相同的模型路径
    output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"
    
    # 加载processor和配置
    processor = AutoProcessor.from_pretrained(output_model_dir, trust_remote_code=True)
    config = UnifiedQwen3Config.from_pretrained(output_model_dir)
    
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
    test_forward()
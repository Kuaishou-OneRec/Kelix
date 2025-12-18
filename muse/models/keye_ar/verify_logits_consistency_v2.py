"""
验证KeyeARModel和KeyeForConditionalGeneration前向logits一致性的脚本
基于test_forward_v2的实现，使用load_keye_ar_config函数初始化KeyeARModel
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'

import json
import torch
import numpy as np
import warnings
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from contextlib import nullcontext

# 导入模型相关模块
from muse.models.keye_ar.ar_ori import KeyeForConditionalGeneration
from muse.models.keye_ar.modeling import KeyeARModel
from muse.models.keye_ar.keye_vl_utils import process_vision_info
from muse.config import KeyeARConfig, UnifiedQwen3Config, KeyeTokenizerConfig, UnifiedTokenDecoderConfig, KeyeVisionConfig


def get_config_value(config_dict, key, default_value, config_name=""):
    """从配置字典中获取值，如果不存在则使用默认值并发出警告"""
    value = config_dict.get(key)
    if value is None:
        value = default_value
        config_source = f" in {config_name}" if config_name else ""
        warnings.warn(f"{key} not found{config_source}, using default value: {default_value}")
    return value


def load_keye_ar_config(conf_path):
    """直接从conf.json加载KeyeARConfig（从test_modeling_forward.py复制）"""
    with open(conf_path, 'r') as f:
        conf_data = json.load(f)
    
    # 构造KeyeVisionConfig
    vision_config_data = conf_data.get('vision_config', {})
    vision_config_inner = vision_config_data.get('vision_config', {})
    
    # 获取vision_config_inner中的字段
    image_size = get_config_value(vision_config_inner, 'image_size', 384, "vision_config")
    patch_size = get_config_value(vision_config_inner, 'patch_size', 14, "vision_config")
    hidden_size = get_config_value(vision_config_inner, 'hidden_size', 1152, "vision_config")
    num_hidden_layers = get_config_value(vision_config_inner, 'num_hidden_layers', 27, "vision_config")
    num_attention_heads = get_config_value(vision_config_inner, 'num_attention_heads', 16, "vision_config")
    intermediate_size = get_config_value(vision_config_inner, 'intermediate_size', 4304, "vision_config")
    hidden_act = get_config_value(vision_config_inner, 'hidden_act', 'gelu_pytorch_tanh', "vision_config")
    layer_norm_eps = get_config_value(vision_config_inner, 'layer_norm_eps', 1e-6, "vision_config")
    attention_dropout = get_config_value(vision_config_inner, 'attention_dropout', 0.0, "vision_config")
    rope_theta = get_config_value(vision_config_inner, 'rope_theta', 10000.0, "vision_config")
    
    keye_vision_config = KeyeVisionConfig(
        model_class="KeyeVL1_5VisionModel",  # 添加model_class字段
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        layer_norm_eps=layer_norm_eps,
        attention_dropout=attention_dropout,
        rope_theta=rope_theta,
    )
    
    # 构造KeyeTokenizerConfig
    codebook_size = get_config_value(vision_config_data, 'codebook_size', 65536, "vision_config_data")
    embedding_dim = get_config_value(vision_config_data, 'embedding_dim', 128, "vision_config_data")
    init_embedding_dim = get_config_value(vision_config_data, 'init_embedding_dim', 4096, "vision_config_data")
    llm_hidden_size = get_config_value(vision_config_data, 'llm_hidden_size', 4096, "vision_config_data")
    n_q_tokens = get_config_value(vision_config_data, 'n_q_tokens', 8, "vision_config_data")
    split_voc = get_config_value(vision_config_data, 'split_voc', 1, "vision_config_data")

    keye_tokenizer_config = KeyeTokenizerConfig(
        model_class="KeyeImageTokenizer",  # 添加model_class字段
        vision_config=keye_vision_config,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        init_embedding_dim=init_embedding_dim,
        llm_hidden_size=llm_hidden_size,
        n_q_tokens=n_q_tokens,
        split_voc=split_voc
    )
    
    print(f"keye_tokenizer_config={keye_tokenizer_config}")
    # 构造UnifiedTokenDecoderConfig
    token_head_dim = get_config_value(conf_data, 'token_head_dim', 512, "conf_data")
    token_head_nhead = get_config_value(conf_data, 'token_head_nhead', 4, "conf_data")
    token_head_intermediate_dim = get_config_value(conf_data, 'token_head_intermediate_dim', 2048, "conf_data")
    token_head_num_layers = get_config_value(conf_data, 'token_head_num_layers', 1, "conf_data")
    hidden_size = get_config_value(conf_data, 'hidden_size', 4096, "conf_data")

    unified_token_decoder_config = UnifiedTokenDecoderConfig(
        model_class="UnifiedTokenDecoder",  # 添加model_class字段
        vocab_size=codebook_size,
        d_model=token_head_dim,
        nhead=token_head_nhead,
        num_layers=1,  # 默认值
        dim_feedforward=token_head_intermediate_dim,
        input_dim=hidden_size,
        reduce=True
    )
    
    # 构造UnifiedQwen3Config
    vocab_size = get_config_value(conf_data, 'vocab_size', 151936, "conf_data")
    
    num_hidden_layers = get_config_value(conf_data, 'num_hidden_layers', 36, "conf_data")
    num_attention_heads = get_config_value(conf_data, 'num_attention_heads', 32, "conf_data")
    num_key_value_heads = get_config_value(conf_data, 'num_key_value_heads', 8, "conf_data")
    head_dim = get_config_value(conf_data, 'head_dim', 128, "conf_data")
    intermediate_size = get_config_value(conf_data, 'intermediate_size', 12288, "conf_data")
    hidden_act = get_config_value(conf_data, 'hidden_act', 'silu', "conf_data")
    rms_norm_eps = get_config_value(conf_data, 'rms_norm_eps', 1e-6, "conf_data")
    attention_dropout = get_config_value(conf_data, 'attention_dropout', 0.0, "conf_data")
    rope_theta = get_config_value(conf_data, 'rope_theta', 1000000, "conf_data")
    max_position_embeddings = get_config_value(conf_data, 'max_position_embeddings', 40960, "conf_data")
    tie_word_embeddings = get_config_value(conf_data, 'tie_word_embeddings', False, "conf_data")

    image_token_id = conf_data.get('image_token_id')
    pad_token_id = conf_data.get('pad_token_id')
    q_eos_token = conf_data.get('q_eos_token')
    
    unified_qwen_config = UnifiedQwen3Config(
        model_class="Qwen3Model",  # 添加model_class字段
        vocab_size=vocab_size,
        embed_dim=hidden_size,
        num_layers=num_hidden_layers,
        num_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_dim=intermediate_size,
        hidden_act=hidden_act,
        norm_eps=rms_norm_eps,
        attn_dropout=attention_dropout,
        tie_word_embeddings=tie_word_embeddings,
        rope_theta=rope_theta,
        max_seq_len=max_position_embeddings,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
        q_eos_token=q_eos_token,
        codebook_size=codebook_size,
        n_q_tokens=n_q_tokens,
        token_head_d_model=token_head_dim,
        token_head_nheads=token_head_nhead,
        token_head_dim_feedforward=token_head_intermediate_dim,
        token_head_num_layers=token_head_num_layers,
    )

    # 构造KeyeARConfig
    keye_ar_config = KeyeARConfig(
        model_class="KeyeARModel",  # 添加model_class字段
        qwen_config=unified_qwen_config,
        tokenizer_config=keye_tokenizer_config,
        token_decoder_config=unified_token_decoder_config,
    )
    
    return keye_ar_config


def generate_circle_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """生成测试用的圆形图像"""
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color, outline=outline_color, width=outline_width)
    return image


def load_keye_for_conditional_generation(output_model_dir, device):
    """加载KeyeForConditionalGeneration模型（ground truth）"""
    model = KeyeForConditionalGeneration.from_pretrained(
        output_model_dir, 
        _attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    model.config.output_one_token = model.output_one_token = False
    model.token_head.use_flash_attn = True
    model = model.to(device).bfloat16()
    return model


def load_keye_ar_model_v2(output_model_dir, device):
    """加载KeyeARModel模型（基于test_forward_v2的实现）"""
    # 加载processor和配置
    processor = AutoProcessor.from_pretrained(output_model_dir, trust_remote_code=True)
    
    # 直接从conf.json加载KeyeARConfig（使用load_keye_ar_config函数）
    config = load_keye_ar_config("muse/models/keye_ar/conf.json")
    
    # 创建模型实例
    model = KeyeARModel(config)
    
    # 从KeyeForConditionalGeneration获取state_dict并转换
    keye_model = KeyeForConditionalGeneration.from_pretrained(
        output_model_dir, 
        _attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    
    # 获取KeyeForConditionalGeneration的state_dict
    keye_state_dict = keye_model.state_dict()
    
    # 转换为KeyeARModel的state_dict
    converted_state_dict = model.convert_hf_state_dict(keye_state_dict, tie_word_embeddings=False)
    model.load_state_dict(converted_state_dict, strict=True)
    
    # 将模型移到设备并转换为bfloat16精度
    model = model.to(device).bfloat16()
    
    return model, processor


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
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def get_keye_conditional_generation_logits(model, inputs):
    """获取KeyeForConditionalGeneration的logits（ground truth）"""
    if torch.cuda.is_available():
        autocast_cm = torch.cuda.amp.autocast
    else:
        try:
            autocast_cm = torch.cpu.amp.autocast
        except Exception:
            autocast_cm = nullcontext

    with autocast_cm(dtype=torch.bfloat16):
        outputs = model(**inputs)
    
    # KeyeForConditionalGeneration直接返回logits
    if hasattr(outputs, 'logits'):
        return outputs.logits
    else:
        # 如果模型输出是tuple，logits通常是第一个元素
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        else:
            raise ValueError("无法获取KeyeForConditionalGeneration的logits")


def get_keye_ar_model_logits(model, inputs):
    """获取KeyeARModel的logits"""
    # 准备KeyeARModel的输入
    inputs_ar = inputs.copy()
    inputs_ar["position_ids"] = torch.arange(0, inputs_ar["input_ids"].size(1)).unsqueeze(0).to(inputs_ar["input_ids"].device)
    inputs_ar["tokens"] = inputs_ar["input_ids"]
    del inputs_ar["input_ids"]
    
    if torch.cuda.is_available():
        autocast_cm = torch.cuda.amp.autocast
    else:
        try:
            autocast_cm = torch.cpu.amp.autocast
        except Exception:
            autocast_cm = nullcontext

    with autocast_cm(dtype=torch.bfloat16):
        outputs = model(**inputs_ar)
    
    import IPython
    IPython.embed()


def compare_logits(logits1, logits2, model1_name, model2_name, tolerance=1e-5):
    """比较两个logits张量是否一致"""
    print(f"\n=== 比较 {model1_name} 和 {model2_name} 的logits ===")
    print(f"Logits1 shape: {logits1.shape}")
    print(f"Logits2 shape: {logits2.shape}")
    
    # 检查形状是否一致
    if logits1.shape != logits2.shape:
        print(f"❌ 形状不一致: {model1_name} {logits1.shape} vs {model2_name} {logits2.shape}")
        return False
    
    # 转换为float32进行精确比较
    logits1_f32 = logits1.float()
    logits2_f32 = logits2.float()
    
    # 计算绝对误差和相对误差
    abs_diff = torch.abs(logits1_f32 - logits2_f32)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    
    # 计算相对误差（避免除以零）
    relative_diff = abs_diff / (torch.abs(logits2_f32) + 1e-8)
    max_relative_diff = torch.max(relative_diff).item()
    mean_relative_diff = torch.mean(relative_diff).item()
    
    print(f"最大绝对误差: {max_abs_diff:.6e}")
    print(f"平均绝对误差: {mean_abs_diff:.6e}")
    print(f"最大相对误差: {max_relative_diff:.6e}")
    print(f"平均相对误差: {mean_relative_diff:.6e}")
    
    # 检查是否在容差范围内
    if max_abs_diff < tolerance and max_relative_diff < tolerance:
        print("✅ Logits完全一致！")
        return True
    else:
        print("❌ Logits不一致！")
        
        # 输出差异最大的位置
        max_diff_indices = torch.argmax(abs_diff.view(-1))
        max_diff_pos = np.unravel_index(max_diff_indices.item(), logits1.shape)
        print(f"最大差异位置: {max_diff_pos}")
        print(f"该位置的值: {model1_name}: {logits1_f32[max_diff_pos]:.6f}, {model2_name}: {logits2_f32[max_diff_pos]:.6f}")
        
        return False


def main():
    """主函数：验证两个模型的logits一致性"""
    # 设置设备
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型路径
    output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"
    
    try:
        # 加载两个模型
        print("正在加载KeyeForConditionalGeneration...")
        keye_conditional_model = load_keye_for_conditional_generation(output_model_dir, device)
        
        print("正在加载KeyeARModel...")
        keye_ar_model, processor = load_keye_ar_model_v2(output_model_dir, device)
        
        # 准备测试消息
        COT_SYSTEM_PROMPT = "You are a helpful assistant."
        messages = [
            {"role": "system",
             "content": [
                 {"type": "text", "text": COT_SYSTEM_PROMPT},
             ], },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": generate_circle_image()},
                    {"type": "text", "text": " What's sum of the first 10 positive integers? After necessary analysis, your final output should follow the format: Final Answer: X."},
                ],
            }
        ]
        
        # 处理输入
        print("处理输入消息...")
        inputs = process_message(messages, processor, device)

        print("获取KeyeARModel的logits...")
        keye_ar_logits = get_keye_ar_model_logits(keye_ar_model, inputs)

        # 获取两个模型的logits
        print("获取KeyeForConditionalGeneration的logits...")
        keye_conditional_logits = get_keye_conditional_generation_logits(keye_conditional_model, inputs)

        
        # 比较logits
        success = compare_logits(
            keye_conditional_logits, 
            keye_ar_logits, 
            "KeyeForConditionalGeneration", 
            "KeyeARModel",
            tolerance=1e-4  # 设置容差
        )
        
        if success:
            print("\n🎉 验证成功！两个模型的前向logits完全一致。")
        else:
            print("\n❌ 验证失败！两个模型的前向logits不一致。")
            return 1
            
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
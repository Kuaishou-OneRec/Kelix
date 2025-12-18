"""
KeyeARModel前向demo脚本
基于test_ar_ori_forward.py修改，适配modeling.py中的KeyeARModel
"""

import os
import json
import torch
import warnings
from pathlib import Path
from transformers import AutoProcessor
from muse.models.keye_ar.modeling import KeyeARModel
from muse.config import KeyeARConfig, UnifiedQwen3Config, KeyeTokenizerConfig, UnifiedTokenDecoderConfig, KeyeVisionConfig
from keye_vl_utils import process_vision_info
from PIL import Image, ImageDraw

# 设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'

def get_config_value(config_dict, key, default_value, config_name=""):
    """从配置字典中获取值，如果不存在则使用默认值并发出警告"""
    value = config_dict.get(key)
    if value is None:
        value = default_value
        config_source = f" in {config_name}" if config_name else ""
        warnings.warn(f"{key} not found{config_source}, using default value: {default_value}")
    return value

def load_keye_ar_config(conf_path):
    """直接从conf.json加载KeyeARConfig"""
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
    
    keye_tokenizer_config = KeyeTokenizerConfig(
        model_class="KeyeImageTokenizer",  # 添加model_class字段
        vision_config=keye_vision_config,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        init_embedding_dim=init_embedding_dim,
        llm_hidden_size=llm_hidden_size,
        n_q_tokens=n_q_tokens,
    )
    
    # 构造UnifiedTokenDecoderConfig
    token_head_dim = get_config_value(conf_data, 'token_head_dim', 512, "conf_data")
    token_head_nhead = get_config_value(conf_data, 'token_head_nhead', 4, "conf_data")
    token_head_intermediate_dim = get_config_value(conf_data, 'token_head_intermediate_dim', 2048, "conf_data")
    token_head_num_layers = get_config_value(conf_data, 'token_head_num_layers', 1, "conf_data")
    
    unified_token_decoder_config = UnifiedTokenDecoderConfig(
        model_class="UnifiedTokenDecoder",  # 添加model_class字段
        vocab_size=codebook_size,
        d_model=token_head_dim,
        nhead=token_head_nhead,
        num_layers=1,  # 默认值
        dim_feedforward=token_head_intermediate_dim,
        input_dim=token_head_dim,
        reduce=True
    )
    
    # 构造UnifiedQwen3Config
    vocab_size = get_config_value(conf_data, 'vocab_size', 151936, "conf_data")
    hidden_size = get_config_value(conf_data, 'hidden_size', 4096, "conf_data")
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

def load_safetensors_state_dict(model_dir):
    """从safetensors文件加载模型状态字典"""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("Please install safetensors package: pip install safetensors")
    
    model_dir = Path(model_dir)
    state_dict = {}
    
    # 查找所有safetensors文件
    safetensors_files = list(model_dir.glob("*.safetensors"))
    
    # 如果有分片的模型文件，加载所有分片
    if safetensors_files:
        for safetensor_file in safetensors_files:
            # 跳过索引文件
            if "index" in str(safetensor_file):
                continue
            print(f"Loading {safetensor_file.name}...")
            shard_state_dict = load_file(str(safetensor_file))
            state_dict.update(shard_state_dict)
    else:
        # 如果没有找到safetensors文件，尝试查找pytorch_model.bin
        pytorch_model_path = model_dir / "pytorch_model.bin"
        if pytorch_model_path.exists():
            print("Loading pytorch_model.bin...")
            state_dict = torch.load(str(pytorch_model_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"No safetensors or pytorch_model.bin files found in {model_dir}")
    
    return state_dict

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
    state_dict = load_safetensors_state_dict(output_model_dir)
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


def generate_circle_image(size=(100, 100), fill_color=(0, 0, 0), outline_color=(255, 255, 255), outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象，用于测试。
    
    Args:
        size: 图像的大小，默认为 (100, 100)
        fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
        outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
        outline_width: 圆的轮廓宽度，默认为 5
        
    Returns:
        生成的 PIL Image 对象
    """
    # 创建一个新的图像对象
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # 计算圆的坐标（图像中心为圆心）
    x_center, y_center = size[0] // 2, size[1] // 2
    radius = min(size[0], size[1]) // 2
    # 绘制圆
    draw.ellipse([x_center - radius, y_center - radius, x_center + radius, y_center + radius],
                 fill=fill_color,
                 outline=outline_color,
                 width=outline_width)
    return image


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
                {"type": "image", "image": generate_circle_image()},
                {"type": "text", "text": " What's sum of the first 10 positive integers? After necessary analysis, your final output should follow the format: Final Answer: X."},
            ],
        }
    ]
    
    inputs = process_message(messages, processor, device)
    inputs["position_ids"] = torch.arange(0, inputs["input_ids"].size(1)).unsqueeze(0).to(device)    
    inputs["tokens"] = inputs["input_ids"]
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
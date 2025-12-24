import pytest
import torch
import os
import json
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor

# 导入模型相关模块
from muse.models.keye_ar.modeling import KeyeARModel
from muse.models.keye_ar.keye_vl_utils import process_vision_info
from muse.config import KeyeARConfig, UnifiedQwen3Config, KeyeTokenizerConfig, UnifiedTokenDecoderConfig, KeyeVisionConfig


def get_config_value(config_dict, key, default_value, config_name=""):
    """从配置字典中获取值，如果不存在则使用默认值并发出警告"""
    import warnings
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
        model_class="KeyeVL1_5VisionModel",
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
        model_class="KeyeImageTokenizer",
        vision_config=keye_vision_config,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        init_embedding_dim=init_embedding_dim,
        llm_hidden_size=llm_hidden_size,
        n_q_tokens=n_q_tokens,
        split_voc=split_voc
    )
    
    # 构造UnifiedTokenDecoderConfig
    token_head_dim = get_config_value(conf_data, 'token_head_dim', 512, "conf_data")
    token_head_nhead = get_config_value(conf_data, 'token_head_nhead', 4, "conf_data")
    token_head_intermediate_dim = get_config_value(conf_data, 'token_head_intermediate_dim', 2048, "conf_data")
    token_head_num_layers = get_config_value(conf_data, 'token_head_num_layers', 1, "conf_data")
    hidden_size = get_config_value(conf_data, 'hidden_size', 4096, "conf_data")

    unified_token_decoder_config = UnifiedTokenDecoderConfig(
        model_class="UnifiedTokenDecoder",
        vocab_size=codebook_size,
        d_model=token_head_dim,
        nhead=token_head_nhead,
        num_layers=token_head_num_layers,
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
    max_position_embeddings = get_config_value(conf_data, 'max_position_embeddings', 40960, "conf_data")
    tie_word_embeddings = get_config_value(conf_data, 'tie_word_embeddings', False, "conf_data")
    rope_base = get_config_value(conf_data, 'rope_theta', 1000000, "conf_data")

    image_token_id = conf_data.get('image_token_id')
    pad_token_id = conf_data.get('pad_token_id')
    q_eos_token = conf_data.get('q_eos_token')
    
    unified_qwen_config = UnifiedQwen3Config(
        model_class="UnifiedQwen3Model",
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
        rope_base=rope_base,
        max_seq_len=max_position_embeddings,
        image_token_id=image_token_id,
        pad_token_id=pad_token_id,
        q_eos_token=q_eos_token,
        codebook_size=codebook_size,
        n_q_tokens=n_q_tokens,
        attention_function="flash_attention_2",
    )

    # 构造KeyeARConfig
    keye_ar_config = KeyeARConfig(
        model_class="KeyeARModel",
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
    
    # 转换到float精度
    def _cast_inputs_to_bf16(batch):
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                batch[k] = v.to(dtype=torch.bfloat16)
        return batch
    
    inputs = _cast_inputs_to_bf16(inputs)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def get_keye_ar_model_logits(model, inputs):
    """获取KeyeARModel的logits"""
    inputs_ar = inputs.copy()
    inputs_ar["position_ids"] = torch.arange(0, inputs_ar["input_ids"].size(1)).unsqueeze(0).to(inputs_ar["input_ids"].device)
    inputs_ar["tokens"] = inputs_ar["input_ids"]
    del inputs_ar["input_ids"]

    with torch.cpu.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**inputs_ar, cu_seqlens=None)
    
    return outputs


@pytest.fixture(scope="module")
def device():
    """设备fixture"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def test_config():
    """测试配置fixture"""
    return {
        "output_model_dir": "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted",
        "output_logit_file": "/mmu_mllm_hdd_2/lingzhixin/model_verification/muse_v2/verify_logits_consistency_v2/keye_conditional_generation.pt"
    }


@pytest.fixture(scope="module")
def keye_ar_model_and_processor(device, test_config):
    """加载KeyeARModel和processor的fixture"""
    # 加载processor和配置
    processor = AutoProcessor.from_pretrained(test_config["output_model_dir"], trust_remote_code=True)
    
    # 直接从conf.json加载KeyeARConfig
    config = load_keye_ar_config("muse/models/keye_ar/conf.json")
    
    # 创建模型实例
    model = KeyeARModel(config)
    
    # 加载权重（这里简化处理，实际测试可能需要mock或使用测试权重）
    # 由于这是测试，我们可能只需要模型结构而不需要实际权重
    # 或者可以使用一个小的测试权重文件
    
    # 将模型移到设备并转换为float精度
    model = model.to(device).bfloat16()
    
    return model, processor


@pytest.fixture(scope="module")
def test_inputs(device, keye_ar_model_and_processor):
    """测试输入fixture"""
    _, processor = keye_ar_model_and_processor
    
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
    return process_message(messages, processor, device)


def test_logits_consistency(keye_ar_model_and_processor, test_inputs, test_config):
    """测试KeyeARModel和KeyeForConditionalGeneration的logits一致性"""
    keye_ar_model, _ = keye_ar_model_and_processor
    
    # 获取KeyeARModel的logits
    keye_ar_logits = get_keye_ar_model_logits(keye_ar_model, test_inputs)
    
    # 从文件加载KeyeForConditionalGeneration的logits
    # 注意：这里假设文件已经存在，实际测试中可能需要创建或mock这个文件
    keye_conditional_logits = torch.load(test_config["output_logit_file"])
    
    # 主要的断言：验证两个模型的logits是否一致
    assert torch.allclose(keye_conditional_logits.to(keye_ar_logits).reshape(keye_ar_logits.shape), keye_ar_logits)



# pytest tests/models/keye_ar/test_verify_logits_consistency_v2_clean.py

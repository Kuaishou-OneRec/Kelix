from transformers import AutoProcessor
from typing import Dict, Any
import torch
import json
import warnings
from pathlib import Path
from muse.config import KeyeARConfig, UnifiedQwen3Config, KeyeTokenizerConfig, UnifiedTokenDecoderConfig, KeyeVisionConfig
from muse.models.keye_ar.modeling import KeyeARModel
from muse.training.common import set_default_dtype


def get_config_value(config_dict, key, default_value, config_name=""):
    """从配置字典中获取值，如果不存在则使用默认值并发出警告"""
    value = config_dict.get(key)
    if value is None:
        value = default_value
        config_source = f" in {config_name}" if config_name else ""
        warnings.warn(f"{key} not found{config_source}, using default value: {default_value}")
    return value


def _build_keye_ar_config(hf_cfg: Dict[str, Any]) -> KeyeARConfig:
    """Map Hugging Face config to Muse KeyeARConfig."""
    
    # 构造KeyeVisionConfig
    vision_config_data = hf_cfg.get('vision_config', {})
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
    token_head_dim = get_config_value(hf_cfg, 'token_head_dim', 512, "hf_cfg")
    token_head_nhead = get_config_value(hf_cfg, 'token_head_nhead', 4, "hf_cfg")
    token_head_intermediate_dim = get_config_value(hf_cfg, 'token_head_intermediate_dim', 2048, "hf_cfg")
    token_head_num_layers = get_config_value(hf_cfg, 'token_head_num_layers', 1, "hf_cfg")
    hidden_size = get_config_value(hf_cfg, 'hidden_size', 4096, "hf_cfg")

    unified_token_decoder_config = UnifiedTokenDecoderConfig(
        model_class="UnifiedTokenDecoder",
        vocab_size=codebook_size,
        d_model=token_head_dim,
        nhead=token_head_nhead,
        num_layers=1,
        dim_feedforward=token_head_intermediate_dim,
        input_dim=hidden_size,
        reduce=True
    )
    
    # 构造UnifiedQwen3Config
    vocab_size = get_config_value(hf_cfg, 'vocab_size', 151936, "hf_cfg")
    
    num_hidden_layers = get_config_value(hf_cfg, 'num_hidden_layers', 36, "hf_cfg")
    num_attention_heads = get_config_value(hf_cfg, 'num_attention_heads', 32, "hf_cfg")
    num_key_value_heads = get_config_value(hf_cfg, 'num_key_value_heads', 8, "hf_cfg")
    head_dim = get_config_value(hf_cfg, 'head_dim', 128, "hf_cfg")
    intermediate_size = get_config_value(hf_cfg, 'intermediate_size', 12288, "hf_cfg")
    hidden_act = get_config_value(hf_cfg, 'hidden_act', 'silu', "hf_cfg")
    rms_norm_eps = get_config_value(hf_cfg, 'rms_norm_eps', 1e-6, "hf_cfg")
    attention_dropout = get_config_value(hf_cfg, 'attention_dropout', 0.0, "hf_cfg")
    rope_theta = get_config_value(hf_cfg, 'rope_theta', 1000000, "hf_cfg")
    max_position_embeddings = get_config_value(hf_cfg, 'max_position_embeddings', 40960, "hf_cfg")
    tie_word_embeddings = get_config_value(hf_cfg, 'tie_word_embeddings', False, "hf_cfg")

    image_token_id = hf_cfg.get('image_token_id')
    pad_token_id = hf_cfg.get('pad_token_id')
    q_eos_token = hf_cfg.get('q_eos_token')
    
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
        base=rope_theta,
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
        model_class="KeyeARModel",
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


def convert_hf_checkpoint(hf_checkpoint_path: str, new_model_dir: str):
    """Convert a Hugging Face Keye checkpoint to a Muse KeyeAR checkpoint"""
    
    # 加载processor
    print(f"Loading processor from {hf_checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(hf_checkpoint_path, trust_remote_code=True)
    
    # 加载Hugging Face配置
    config_path = Path(hf_checkpoint_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        hf_config_dict = json.load(f)
    
    # 构建Muse配置
    print("Building Muse configuration...")
    config = _build_keye_ar_config(hf_config_dict)
    
    # 创建Muse模型实例
    print("Creating Muse model instance...")
    with set_default_dtype(torch.bfloat16):
        model = KeyeARModel(config)
    
    # 加载Hugging Face权重
    print("Loading Hugging Face weights...")
    hf_state_dict = load_safetensors_state_dict(hf_checkpoint_path)
    
    # 转换权重
    print("Converting weights to Muse format...")
    converted_state_dict = model.convert_hf_state_dict(hf_state_dict, tie_word_embeddings=False)
    
    # 加载权重到Muse模型
    print("Loading converted weights...")
    model.load_state_dict(converted_state_dict, strict=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # 将模型移到设备并设置精度
    model = model.to(device=device, dtype=dtype)
    
    # 检查参数精度
    for name, param in model.named_parameters():
        if param.dtype != dtype:
            print(f"Warning: Parameter {name} has dtype {param.dtype}, expected {dtype}")
            param.data = param.data.to(dtype=dtype)
    
    for name, buffer in model.named_buffers():
        if buffer.dtype != dtype:
            print(f"Warning: Buffer {name} has dtype {buffer.dtype}, expected {dtype}")
            buffer.data = buffer.data.to(dtype=dtype)
    
    print("Model conversion completed successfully!")
    
    # 保存转换后的模型
    print(f"Saving converted model to {new_model_dir}...")
    model.save_pretrained(new_model_dir)
    processor.save_pretrained(new_model_dir)
    
    print("✓✓✓ SUCCESS: KeyeAR model conversion completed!")


if __name__ == "__main__":
    # 使用指定的路径进行转换
    output_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/converted"
    new_model_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step4000/global_step4000/muse_converted"
    
    convert_hf_checkpoint(output_model_dir, new_model_dir)
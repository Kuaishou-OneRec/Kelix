from typing import Dict, Any, Tuple
from pathlib import Path
import json
import argparse
import torch
from PIL import Image, ImageDraw
from muse.config import KeyeVisionConfig, KeyeTokenizerConfig
from muse.models.keye_tokenizer import KeyeImageTokenizer
from muse.training.common import set_default_dtype
from muse.training.checkpoint import load_hf_checkpoint

def _build_muse_tokenizer_config(hf_config: Dict[str, Any]) -> KeyeTokenizerConfig:
    """Build Muse KeyeTokenizerConfig from raw config dictionary."""
    outer_vcfg = hf_config["vision_config"]
    inner_vcfg = outer_vcfg["vision_config"]
    
    vision_cfg = KeyeVisionConfig(
        hidden_size=inner_vcfg["hidden_size"],
        num_hidden_layers=inner_vcfg["num_hidden_layers"],
        num_attention_heads=inner_vcfg["num_attention_heads"],
        image_size=inner_vcfg["image_size"],
        patch_size=inner_vcfg["patch_size"],
        intermediate_size=inner_vcfg["intermediate_size"],
        hidden_act=inner_vcfg.get("hidden_act", "gelu_pytorch_tanh"),
        has_learnable_position_embedding=inner_vcfg.get("has_learnable_position_embedding", True),
        attention_dropout=inner_vcfg.get("attention_dropout", 0.0),
        rope_theta=inner_vcfg.get("rope_theta", 10000.0),
        use_qk_norm=inner_vcfg.get("use_qk_norm", False),
        qk_norm_eps=inner_vcfg.get("qk_norm_eps", 1e-6),
        attention_function=hf_config.get("_attn_implementation", "flash_attention_2"),
    )
    
    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=outer_vcfg.get("llm_hidden_size", 4096),
        embedding_dim=outer_vcfg.get("embedding_dim", 128),
        init_embedding_dim=outer_vcfg.get("init_embedding_dim", 4096),
        codebook_size=outer_vcfg.get("codebook_size", 65536),
        n_q_tokens=outer_vcfg.get("n_q_tokens", 8),
        split_voc=outer_vcfg.get("split_voc", 1),
        add_voc_reducer=outer_vcfg.get("add_voc_reducer", False),
        split_dim=outer_vcfg.get("split_dim", False),
        vq_sampling_mode="argmin",
        vq_temperature=1.0,
        vq_temperature_decay=0.999,
        vq_min_temperature=0.1,
    )
    return tokenizer_cfg

def convert_hf_checkpoint(hf_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract visual_tokenizer weights from full Keye-VL checkpoint,
    convert to Muse format, and save to local path.
    
    Returns the converted Muse-style state dict.
    """
    # save_dir = Path(save_path)
    # save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract visual_tokenizer.* keys from full state dict
    origin_tokenizer_state_dict = {}
    for k, v in hf_state_dict.items():
        if k.startswith("visual_tokenizer."):
            new_k = k[len("visual_tokenizer."):]
            origin_tokenizer_state_dict[new_k] = v

    muse_state_dict = {}
    for k, v in origin_tokenizer_state_dict.items():
        new_k = k
        
        # Convert visual.vision_model.* to visual.*
        if k.startswith("visual.vision_model."):
            new_k = "visual." + k[len("visual.vision_model."):]
        
        # Convert encoder.layers.X.layer_norm1 -> encoder.layers.X.sa_norm
        new_k = new_k.replace(".layer_norm1.", ".sa_norm.")
        # Convert encoder.layers.X.layer_norm2 -> encoder.layers.X.mlp_norm
        new_k = new_k.replace(".layer_norm2.", ".mlp_norm.")
        # Convert self_attn -> attn
        new_k = new_k.replace(".self_attn.", ".attn.")
        # Convert out_proj -> output_proj
        new_k = new_k.replace(".out_proj.", ".output_proj.")
        # Convert mlp.fc1 -> mlp.w1
        new_k = new_k.replace(".mlp.fc1.", ".mlp.w1.")
        # Convert mlp.fc2 -> mlp.w2
        new_k = new_k.replace(".mlp.fc2.", ".mlp.w2.")
        # Convert post_layernorm -> ln_post
        new_k = new_k.replace(".post_layernorm.", ".ln_post.")
        
        muse_state_dict[new_k] = v
    
    return muse_state_dict


def load_muse_tokenizer_from_saved(
    save_path: str,
    device: str,
    dtype: torch.dtype,
) -> Tuple[KeyeImageTokenizer, KeyeTokenizerConfig]:
    """
    Load Muse KeyeImageTokenizer from saved weights.
    """
    save_dir = Path(save_path)
    
    # Load config
    config_path = save_dir / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Build KeyeTokenizerConfig from saved config
    vision_cfg_dict = config_dict.get("vision_config", {})
    vision_cfg = KeyeVisionConfig(**vision_cfg_dict)
    
    tokenizer_cfg = KeyeTokenizerConfig(
        vision_config=vision_cfg,
        llm_hidden_size=config_dict.get("llm_hidden_size", 4096),
        embedding_dim=config_dict.get("embedding_dim", 128),
        init_embedding_dim=config_dict.get("init_embedding_dim", 4096),
        codebook_size=config_dict.get("codebook_size", 65536),
        n_q_tokens=config_dict.get("n_q_tokens", 8),
        split_voc=config_dict.get("split_voc", 1),
        add_voc_reducer=config_dict.get("add_voc_reducer", False),
        split_dim=config_dict.get("split_dim", False),
        vq_sampling_mode=config_dict.get("vq_sampling_mode", "argmin"),
        vq_temperature=config_dict.get("vq_temperature", 1.0),
        vq_temperature_decay=config_dict.get("vq_temperature_decay", 0.999),
        vq_min_temperature=config_dict.get("vq_min_temperature", 0.1),
    )
    
    # Initialize model
    with set_default_dtype(dtype):
        muse_tokenizer = MuseKeyeImageTokenizer(tokenizer_cfg).to(device)
    
    # Load weights
    weights_path = save_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location="cpu")
    
    missing, unexpected = muse_tokenizer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Muse tokenizer missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"Muse tokenizer unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
    
    muse_tokenizer.eval()
    print(f"Loaded Muse tokenizer from: {save_path}")
    
    return muse_tokenizer, tokenizer_cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--processor-dir", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()

def main():
    args = get_args()
    hf_config_path = Path(args.hf_dir) / "config.json"
    with open(hf_config_path) as f:
        hf_config = json.loads(f.read())
    config = _build_muse_tokenizer_config(hf_config)
    hf_state_dict = load_hf_checkpoint(args.hf_dir)
    state_dict = convert_hf_checkpoint(hf_state_dict)

    with set_default_dtype(args.dtype), torch.device("cpu"):
        tokenizer = KeyeImageTokenizer(config)

    missing, unexpected = tokenizer.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Tokenizer missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  - {k}")
    if unexpected:
        print(f"Tokenizer unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  - {k}")
    
    tokenizer.save_pretrained(args.output_dir)

def generate_circle_image(
        size=(100, 100),
        fill_color=(0, 0, 0),
        outline_color=(255, 255, 255),
        outline_width=5):
    """
    生成一个包含一个圆的 PIL Image 对象，用于测试。
    
    :param size: 图像的大小，默认为 (64, 64)
    :param fill_color: 圆的填充颜色，默认为黑色 (0, 0, 0)
    :param outline_color: 圆的轮廓颜色，默认为白色 (255, 255, 255)
    :param outline_width: 圆的轮廓宽度，默认为 5
    :return: 生成的 PIL Image 对象
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


def process_message(messages, device):
    return inputs

def test_demo():
    with set_default_dtype("bfloat16"), torch.device("cuda"):
        tokenizer = KeyeImageTokenizer.from_pretrained(
            "/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer/"
        )
    
    
    from transformers import AutoProcessor
    from keye_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(
        "/llm_reco_ssd/zhouyang12/models/muse/KeyeTokenizer",
        trust_remote_code=True
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": generate_circle_image()},
        ],
    }]
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True  # 开启生成提示
    )

    image_inputs, _, _ = process_vision_info(messages)

    # 构建原始输入（纯有效Token，无任何Pad）
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=False,  # 强制关闭Pad，确保原始输入无多余Token
        truncation=False,
        return_tensors="pt",
    ).to("cuda")

    dumps = torch.load("/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/debug_new/vq_outputs.pt")
    dumps2 = torch.load("/mmu_mllm_hdd_2/zhouyang12/output/Keye/vq_end2end_1105/run_exp1.6.6109_stage3/step9500/global_step9500/converted/debug_new/test_run_outputs.pt")

    torch.testing.assert_allclose(inputs["pixel_values"].cpu(), dumps["pixel_values"])
    torch.testing.assert_allclose(inputs["image_grid_thw"].cpu(), dumps["image_grid_thw"])

    with torch.no_grad():
        vq_out = tokenizer(pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
    
    # print(dumps.keys())
    torch.testing.assert_allclose(vq_out["x"].cpu(), dumps["image_embeds"])

    indices = torch.stack([x_i for x_i in vq_out['indices']], 0).T 
    aligned_indices = 151936 + indices + torch.arange(8).\
        to("cuda")[None] * tokenizer.config.codebook_size // 8

    answer = torch.LongTensor(
        [[157696, 161428, 172176, 176543, 191772, 198720, 201995, 209754],
        [155872, 161428, 172475, 178883, 189182, 195380, 203676, 215606],
        [152707, 162232, 170817, 177567, 190989, 194128, 203905, 211502],
        [155872, 161428, 169345, 181246, 186400, 193588, 203676, 210683],
        [156170, 164215, 175621, 180185, 192150, 197081, 208022, 216938],
        [153068, 160320, 172737, 178954, 185988, 198887, 203676, 212428],
        [152707, 167798, 174112, 182049, 187728, 193269, 206012, 214532],
        [156614, 166166, 172568, 184363, 189079, 199763, 207621, 212810],
        [152707, 160713, 172989, 181146, 192749, 195380, 207734, 215305],
        [159674, 166192, 171391, 181686, 190231, 195874, 203905, 215606],
        [152707, 161925, 174112, 182049, 190989, 195761, 201154, 214532],
        [156614, 167079, 172568, 182049, 185202, 195874, 206012, 214334],
        [156614, 160645, 175321, 181067, 189079, 199763, 202266, 212062],
        [153068, 160645, 172568, 178561, 187447, 199763, 207621, 214334],
        [152707, 167053, 174112, 182049, 187728, 195874, 201204, 214532],
        [153068, 162232, 171391, 178561, 192513, 195874, 204058, 210162]]
    ).to("cuda")
    torch.testing.assert_allclose(aligned_indices, answer)

if __name__ == "__main__":
    test_demo()

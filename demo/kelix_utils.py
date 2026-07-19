#!/usr/bin/env python3
"""
Shared utilities for the Kelix demos.

Encapsulates model loading (Kelix-Tok + Kelix-LLM, Kelix-DiT, DC-AE VAE) and
the text-to-image pipeline so that `demo_kelix.py` and `demo_kelix_t2i.py`
stay short and focused on the core flow: load model -> write prompt -> generate.

All paths are overridable via environment variables:
  KELIX_DIR               Kelix AR model (Tok + LLM) dir
  DIT_DIR                 Kelix-DiT de-tokenizer dir
  VAE_DIR                 Frozen DC-AE VAE dir
  MODEL_CONFIG_OVERRIDES  DiT config overrides (default: "model_max_length=720")
"""

import os
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from diffusers import FlowMatchEulerDiscreteScheduler

from muse.config import load_config
from muse.models import get_model_class
from muse.models.keye_ar import KeyeARModel
from muse.training.common import set_default_dtype, get_torch_dtype
from muse.training.checkpoint import load_hf_checkpoint
from muse.utils.common import parse_config_overrides
from recipes.sana.utils import load_vae
from recipes.sana.train_sana_ar_dit import compute_pos_args
from recipes.sana.inference_ar2image import tokenize_images


def _resolve_dir(path: str) -> str:
    """Resolve a model directory: use it directly if it's a local dir,
    otherwise download from HuggingFace Hub (treating `path` as a repo id).

    Supports repo ids with an optional subdirectory, e.g.
    ``namespace/repo_name/sub/dir`` — the full repo is downloaded, then the
    subdirectory path is appended. This is needed for the VAE which lives in
    a subfolder of the SANA1.5 diffusers repo.

    This lets the demos run out-of-the-box with the default HF repo ids
    (e.g. ``OpenOneRec/Kelix-SFT``) without leaking any internal paths,
    while still accepting a local directory via the env vars.
    """
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download
    print(f"[kelix] '{path}' is not a local dir; downloading from HuggingFace Hub...")
    parts = path.split("/")
    if len(parts) <= 2:
        return snapshot_download(repo_id=path)
    repo_id = "/".join(parts[:2])
    sub_dir = "/".join(parts[2:])
    local_root = snapshot_download(
        repo_id=repo_id, allow_patterns=[f"{sub_dir}/*"]
    )
    return os.path.join(local_root, sub_dir)

# ---------------------------------------------------------------------------
# Config (overridable via env vars)
# ---------------------------------------------------------------------------
KELIX_DIR = os.environ.get(
    "KELIX_DIR",
    "OpenOneRec/Kelix-SFT",
)
DIT_DIR = os.environ.get(
    "DIT_DIR",
    "OpenOneRec/Kelix-DiT",
)
# Frozen DC-AE VAE from the SANA1.5 release.
#   https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/tree/main/vae
VAE_DIR = os.environ.get(
    "VAE_DIR",
    "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/vae",
)
DEVICE = "cuda:0"
DTYPE_STR = "bfloat16"

# DiT sampling config.
IMAGE_SIZE = 1024
MAX_CONDITION_LENGTH = 720
NUM_SAMPLING_STEPS = 50
CFG_SCALE = 1.0
FLOW_SHIFT = 3.0
SEED = 42

# DiT model-config overrides. The released Kelix-DiT checkpoint was trained
# with model_max_length=720; the config.json default (300) must be overridden
# to match, otherwise y_embedder.y_embedding shape mismatches on load_state_dict.
MODEL_CONFIG_OVERRIDES = tuple(
    filter(None, os.environ.get("MODEL_CONFIG_OVERRIDES", "model_max_length=720").split(","))
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the configured device (falls back to CPU if no CUDA)."""
    return torch.device(DEVICE if torch.cuda.is_available() else "cpu")


def get_dtype() -> torch.dtype:
    """Return the configured torch dtype."""
    return get_torch_dtype(DTYPE_STR)


def load_ar_model(
    ar_dir: str = KELIX_DIR,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> "KeyeARModel":
    """Load the Kelix AR model (Kelix-Tok + Kelix-LLM) in local (non-distributed) mode."""
    device = device or get_device()
    dtype = dtype or get_dtype()
    ar_dir = _resolve_dir(ar_dir)
    with set_default_dtype(dtype), torch.device(device):
        ar = KeyeARModel.from_pretrained(ar_dir).eval()  # type: ignore[assignment]
    ar.config.qwen_config.output_last_hidden_states_only = False
    ar.model.model.output_last_hidden_states_only = False  # type: ignore[attr-defined]
    ar.requires_grad_(False)
    return ar  # type: ignore[return-value]


def load_processor(ar_dir: str = KELIX_DIR) -> AutoProcessor:
    """Load the AutoProcessor (chat template + tokenizer + image processor)."""
    ar_dir = _resolve_dir(ar_dir)
    return AutoProcessor.from_pretrained(ar_dir, trust_remote_code=True)


def load_dit(
    model_dir: str = DIT_DIR,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    """Load the Kelix-DiT de-tokenizer from a converted checkpoint dir."""
    device = device or get_device()
    dtype = dtype or get_dtype()
    model_dir = _resolve_dir(model_dir)
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"DiT config not found at {cfg_path}")
    model_config = load_config(cfg_path)

    # Apply config overrides (e.g. model_max_length=720) to match the checkpoint.
    if MODEL_CONFIG_OVERRIDES:
        overrides = parse_config_overrides(list(MODEL_CONFIG_OVERRIDES))
        print(f"[kelix] DiT config overrides: {overrides}")
        for k, v in overrides.items():
            if hasattr(model_config, k):
                print(f"[kelix]   {k}: {getattr(model_config, k)} -> {v}")
                setattr(model_config, k, v)
            else:
                raise ValueError(f"Unknown DiT model config field: {k}")

    model_cls = get_model_class(model_config.model_class)
    with set_default_dtype(dtype), torch.device("cpu"):
        dit = model_cls(model_config)
    sd = load_hf_checkpoint(model_dir)
    dit.load_state_dict(sd, strict=False)
    dit.to(device).to(dtype=dtype)
    dit.eval()
    return dit


def load_vae_model(
    vae_dir: str = VAE_DIR,
    device: torch.device = None,
    dtype: torch.dtype = None,
):
    """Load the frozen DC-AE VAE."""
    device = device or get_device()
    dtype = dtype or get_dtype()
    vae_dir = _resolve_dir(vae_dir)
    return load_vae(vae_dir, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Input building
# ---------------------------------------------------------------------------

def make_test_image(size: int = 224) -> Image.Image:
    """Draw a simple black circle on white background (for the understanding demo)."""
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 3
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0))
    return img


def process_message(
    processor: AutoProcessor,
    device: torch.device,
    messages: list,
    add_generation_prompt: bool = True,
):
    """Apply chat template + process vision info -> model input dict."""
    from keye_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    image_inputs, video_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    return inputs


def build_input_ids(processor: AutoProcessor, device: torch.device, prompt: str):
    """Build model input_ids from a text prompt via the chat template."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(text=[text], return_tensors="pt").to(device)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def chat(
    model: "KeyeARModel",
    processor: AutoProcessor,
    messages: list,
    max_new_tokens: int = 256,
    top_k: int = 1,
) -> str:
    """Run a multi-turn chat: messages -> text response.

    `messages` follows the OpenAI-style format, e.g.:
        [{"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "What's in the image?"},
        ]}]
    """
    device = next(model.parameters()).device
    inputs = process_message(processor, device, messages)
    output_ids = model.generate(**inputs, top_k=top_k, max_new_tokens=max_new_tokens)
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    # skip_special_tokens=False so <|vision_start|>/<|vision_end|>/<|im_end|> are
    # preserved in the decoded text (needed to inspect generation structure).
    return processor.decode(new_ids[:, 0].long().tolist(), skip_special_tokens=False)


@torch.no_grad()
def generate_image_tokens(
    model: "KeyeARModel",
    processor: AutoProcessor,
    prompt: str,
    max_new_tokens: int = 720,
    top_k: int = 1,
):
    """Generate discrete image tokens from a text prompt.

    Returns (output_ids, content, image_token_groups) where:
      - output_ids: raw generated token tensor (seq_len, n_tokens)
      - content: decoded text (special tokens preserved)
      - image_token_groups: list from model.extract_image_tokens
    """
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = process_message(processor, device, messages)
    output_ids = model.generate(**inputs, top_k=top_k, max_new_tokens=max_new_tokens)
    output_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    content = processor.decode(output_ids[:, 0].long().tolist(), skip_special_tokens=False)
    image_token_groups = model.extract_image_tokens(output_ids)
    return output_ids, content, image_token_groups


@torch.no_grad()
def generate_image(
    ar_model: "KeyeARModel",
    ar_processor: AutoProcessor,
    dit,
    vae,
    prompt: str,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Image.Image:
    """Text-to-image: prompt -> 1024x1024 PIL.Image.

    Pipeline:
      prompt -> AR model generates hidden states -> DiT flow-matching sampling
      -> VAE decode -> PIL.Image.
    """
    device = device or get_device()
    dtype = dtype or get_dtype()

    # 1. prompt -> input_ids
    inputs = build_input_ids(ar_processor, device, prompt)
    print(f"[kelix] input_ids shape: {tuple(inputs['input_ids'].shape)}")

    # 2. AR model generates image tokens + last hidden states; extract cond embeddings.
    cond_embeds, cond_mask, token_embed_lengths = tokenize_images(  # type: ignore[misc]
        ar_processor=ar_processor,
        ar_model=ar_model,
        batch_size=1,
        max_condition_length=MAX_CONDITION_LENGTH,
        input_ids=inputs["input_ids"],
        teacher_forcing=False,
        condition_on_special_tokens=True,
    )
    print(
        f"[kelix] cond_embeds: {tuple(cond_embeds.shape)}  "
        f"cond_mask: {tuple(cond_mask.shape)}"
    )

    # 3. Project condition embeddings through DiT's diffusion connector.
    cond_embeds = dit.diffusion_connector(cond_embeds)

    # 4. Build unconditional embeddings for CFG.
    null_embed = dit.y_embedder.y_embedding
    seq_len = min(null_embed.shape[0], MAX_CONDITION_LENGTH)
    uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(1, -1, -1)
    if seq_len < MAX_CONDITION_LENGTH:
        padding = torch.zeros(
            1, MAX_CONDITION_LENGTH - seq_len, uncond_embeds.shape[-1],
            device=device, dtype=dtype,
        )
        uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
    uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
    uncond_mask = torch.zeros(1, MAX_CONDITION_LENGTH, device=device)
    uncond_mask[:, :seq_len] = 1
    uncond_mask = uncond_mask[:, None, None, :]

    # 5. Flow-matching sampling loop.
    latent_channels = vae.config.latent_channels
    latent_size = IMAGE_SIZE // dit.config.vae_downsample_rate

    scheduler = FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT)
    sigmas = np.linspace(1.0, 1 / NUM_SAMPLING_STEPS, NUM_SAMPLING_STEPS)
    scheduler.set_timesteps(NUM_SAMPLING_STEPS, sigmas=sigmas, device=device)

    generator = torch.Generator(device=device).manual_seed(SEED)
    dit_latents = torch.randn(
        (1, latent_channels, latent_size, latent_size),
        generator=generator, device=device, dtype=dtype,
    )

    cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
    mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)

    pos_args = compute_pos_args(
        latent_hw=(latent_size, latent_size),
        image_grid_thw=torch.tensor(
            [1, 2 * MAX_CONDITION_LENGTH**0.5, 2 * MAX_CONDITION_LENGTH**0.5]
        )[None].to(device),
        max_seq_len=MAX_CONDITION_LENGTH,
        device=device,
        cond_pos_scale=1.0,
        image_size=IMAGE_SIZE,
        token_embed_lengths=token_embed_lengths,
    )
    model_kwargs = {**pos_args, "is_y_connected": True}

    for i, t in enumerate(scheduler.timesteps):
        latent_input = torch.cat([dit_latents] * 2)
        timestep = t.expand(latent_input.shape[0])
        noise_pred = dit.forward_with_dpmsolver(
            latent_input, timestep, cond_embeds_cfg, mask=mask_cfg, **model_kwargs
        )
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + CFG_SCALE * (noise_cond - noise_uncond)
        dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]
        if (i + 1) % 10 == 0:
            print(f"[kelix] sampling step {i + 1}/{NUM_SAMPLING_STEPS}")

    # 6. VAE decode -> PIL image.
    recon_latents = dit_latents / vae.config.scaling_factor
    images = vae.decode(recon_latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    img_np = images[0].detach().cpu().permute(1, 2, 0).float().numpy()
    return Image.fromarray((img_np * 255).round().astype("uint8"))

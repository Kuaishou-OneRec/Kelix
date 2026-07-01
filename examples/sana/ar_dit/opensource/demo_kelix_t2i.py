#!/usr/bin/env python3
"""
Kelix text-to-image demo: prompt -> 1024x1024 image.

Loads three components:
  1. Kelix AR model  (Kelix-Tok + Kelix-LLM)  — from KELIX_DIR
  2. Kelix-DiT       (diffusion de-tokenizer)  — from DIT_DIR
  3. DC-AE VAE       (frozen latent decoder)   — from VAE_DIR
     (default: /llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/)

Pipeline:
  prompt
    -> chat-template input_ids
    -> AR model generates image tokens + last hidden states (via tokenize_images)
    -> DiT diffusion_connector projects hidden states
    -> flow-matching sampling in VAE latent space (32x32)
    -> VAE decode -> PIL.Image (1024x1024)

Reference: examples/keye_ar/serve_dit/serve_visualize_reconstruction.py

Usage:
    python examples/sana/ar_dit/opensource/demo_kelix_t2i.py
    python examples/sana/ar_dit/opensource/demo_kelix_t2i.py --prompt "Draw a cat in comic style"
    python examples/sana/ar_dit/opensource/demo_kelix_t2i.py --prompt "a comic-style cat" --output out.png

    # Override paths via env vars:
    KELIX_DIR=/path/to/release_sft DIT_DIR=/path/to/release_dit \
    VAE_DIR=/path/to/vae python examples/sana/ar_dit/opensource/demo_kelix_t2i.py
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from diffusers import FlowMatchEulerDiscreteScheduler

from muse.config import load_config
from muse.models import get_model_class
from muse.models.keye_ar import KeyeARModel
from muse.training.common import set_default_dtype, get_torch_dtype
from muse.training.checkpoint import load_hf_checkpoint
from recipes.sana.utils import load_vae
from recipes.sana.train_sana_ar_dit import compute_pos_args
from recipes.sana.inference_ar2image import tokenize_images

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KELIX_DIR = os.environ.get(
    "KELIX_DIR",
    "/mmu_mllm_hdd_2/lingzhixin/output/release/muse/release_sft",
)
DIT_DIR = os.environ.get(
    "DIT_DIR",
    "/mmu_mllm_hdd_2/lingzhixin/output/release/muse/release_dit",
)
# The frozen DC-AE VAE ships with the SANA1.5 release and is loaded as-is.
#   https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/tree/main/vae
# The default below is the canonical local mirror used across the repo's
# training/inference scripts.
VAE_DIR = os.environ.get(
    "VAE_DIR",
    "/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/",
)
DEVICE = "cuda:0"
DTYPE_STR = "bfloat16"

IMAGE_SIZE = 1024
MAX_CONDITION_LENGTH = 720
NUM_SAMPLING_STEPS = 50
CFG_SCALE = 1.0
FLOW_SHIFT = 3.0
SEED = 42


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ar_model(ar_dir: str, device: torch.device, dtype: torch.dtype) -> KeyeARModel:
    """Load the Kelix AR model (tokenizer + LLM) in local (non-distributed) mode."""
    with set_default_dtype(dtype), torch.device(device):
        ar = KeyeARModel.from_pretrained(ar_dir).eval()  # type: ignore[assignment]
    ar.config.qwen_config.output_last_hidden_states_only = False
    ar.model.model.output_last_hidden_states_only = False  # type: ignore[attr-defined]
    ar.requires_grad_(False)
    return ar  # type: ignore[return-value]


def load_dit(model_dir: str, device: torch.device, dtype: torch.dtype):
    """Load the Kelix-DiT de-tokenizer from a converted checkpoint dir."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"DiT config not found at {cfg_path}")
    model_config = load_config(cfg_path)
    model_cls = get_model_class(model_config.model_class)
    with set_default_dtype(dtype), torch.device("cpu"):
        dit = model_cls(model_config)
    sd = load_hf_checkpoint(model_dir)
    dit.load_state_dict(sd, strict=False)
    dit.to(device).to(dtype=dtype)
    dit.eval()
    return dit


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_input_ids(processor, device, prompt: str):
    """Build model input_ids from a text prompt via the chat template."""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to(device)
    return inputs


@torch.no_grad()
def generate_image(
    ar_model: KeyeARModel,
    ar_processor: AutoProcessor,
    dit,
    vae,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Image.Image:
    """prompt -> PIL.Image (1024x1024)."""

    # 1. prompt -> input_ids
    inputs = build_input_ids(ar_processor, device, prompt)
    print(f"[demo] input_ids shape: {tuple(inputs['input_ids'].shape)}")

    # 2. AR model generates image tokens + last hidden states; extract cond embeddings.
    # tokenize_images returns [cond_embeds, cond_mask, token_embed_lengths].
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
        f"[demo] cond_embeds: {tuple(cond_embeds.shape)}  "
        f"cond_mask: {tuple(cond_mask.shape)}  "
        f"token_embed_lengths: {token_embed_lengths}"
    )

    # 3. Project condition embeddings through DiT's diffusion connector.
    cond_embeds = dit.diffusion_connector(cond_embeds)

    # 4. Build unconditional embeddings for CFG (classifier-free guidance).
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
            print(f"[demo] sampling step {i + 1}/{NUM_SAMPLING_STEPS}")

    # 6. VAE decode -> PIL image.
    recon_latents = dit_latents / vae.config.scaling_factor
    images = vae.decode(recon_latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    img_np = images[0].detach().cpu().permute(1, 2, 0).float().numpy()
    return Image.fromarray((img_np * 255).round().astype("uint8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kelix text-to-image demo")
    parser.add_argument("--prompt", type=str, default="Draw a cat in comic style")
    parser.add_argument("--output", type=str, default="./kelix_t2i_demo.png")
    args = parser.parse_args()

    dtype = get_torch_dtype(DTYPE_STR)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print(f"Loading AR model from {KELIX_DIR} ...")
    ar_model = load_ar_model(KELIX_DIR, device, dtype)
    ar_processor = AutoProcessor.from_pretrained(KELIX_DIR, trust_remote_code=True)

    print(f"Loading DiT from {DIT_DIR} ...")
    dit = load_dit(DIT_DIR, device, dtype)

    print(f"Loading VAE from {VAE_DIR} ...")
    vae = load_vae(VAE_DIR, device=device, dtype=dtype)

    print(f"\nPrompt: {args.prompt}")
    img = generate_image(ar_model, ar_processor, dit, vae, args.prompt, device, dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    img.save(args.output, quality=95)
    print(f"\nSaved: {args.output}  ({img.size})")


if __name__ == "__main__":
    main()

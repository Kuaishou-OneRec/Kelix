#!/usr/bin/env python3
"""
Kelix demo: image understanding + image-token generation.

Loads the Kelix unified model (Kelix-Tok + Kelix-LLM, i.e. KeyeARModel) from a
local checkpoint directory and runs two simple tasks:

  1. Image understanding  — feed an image + question, get a text answer.
  2. Image-token generation — give a text prompt, generate discrete visual
     tokens (which can later be fed to Kelix-DiT for pixel rendering, or
     fed back to the model for image-grounded QA).

Reference: tests/models/keye_ar/test_keyear_muse_forward.py

Usage:
    python examples/sana/ar_dit/opensource/demo_kelix.py

    # Override the checkpoint dir:
    KELIX_DIR=/path/to/release_sft python examples/sana/ar_dit/opensource/demo_kelix.py
"""

import os

import torch
from transformers import AutoProcessor
from keye_vl_utils import process_vision_info
from PIL import Image, ImageDraw

from muse.models.keye_ar import KeyeARModel
from muse.training.common import set_default_dtype

# ---------------------------------------------------------------------------
# Config — adjust KELIX_DIR to point at your local Kelix-SFT release folder.
# ---------------------------------------------------------------------------
KELIX_DIR = os.environ.get(
    "KELIX_DIR",
    "/mmu_mllm_hdd_2/lingzhixin/output/release/muse/release_sft",
)
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_image(size: int = 224) -> Image.Image:
    """Draw a simple black circle on white background for the understanding demo."""
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 3
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0))
    return img


def process_message(processor, device, messages, add_generation_prompt: bool = True):
    """Apply chat template + process vision info -> model input dict."""
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


# ---------------------------------------------------------------------------
# Demo tasks
# ---------------------------------------------------------------------------

def demo_understanding(model, processor):
    """Task 1: image understanding — ask a question about an image."""
    print("\n" + "=" * 60)
    print("Task 1: Image Understanding")
    print("=" * 60)

    image = make_test_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What's in the image? Describe it briefly."},
            ],
        }
    ]
    inputs = process_message(processor, next(model.parameters()).device, messages)
    output_ids = model.generate(**inputs, top_k=1, max_new_tokens=256)
    # Strip the prompt tokens.
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    content = processor.decode(new_ids[:, 0].long().tolist())
    print(f"User : What's in the image?")
    print(f"Kelix: {content}")


def demo_generation(model, processor):
    """Task 2: image-token generation — generate discrete visual tokens from a prompt."""
    print("\n" + "=" * 60)
    print("Task 2: Image-Token Generation")
    print("=" * 60)

    prompt = "Generate an image of a cute cat."
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    inputs = process_message(processor, next(model.parameters()).device, messages)
    output_ids = model.generate(**inputs, top_k=1, max_new_tokens=450)
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    content = processor.decode(new_ids[:, 0].long().tolist())
    print(f"User : {prompt}")
    print(f"Kelix (raw tokens): {content}")

    # Extract the discrete image-token groups for downstream use (e.g. feeding
    # to Kelix-DiT, or to a follow-up understanding turn via fill_image_tokens).
    image_token_groups = model.extract_image_tokens(new_ids)
    print(
        f"Extracted {len(image_token_groups)} image-token group(s), "
        f"shapes: {[x.shape for x in image_token_groups]}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print(f"Loading Kelix from {KELIX_DIR} ...")
    processor = AutoProcessor.from_pretrained(KELIX_DIR, trust_remote_code=True)
    with set_default_dtype(DTYPE):
        model = KeyeARModel.from_pretrained(KELIX_DIR).to(DEVICE)
    model.eval()

    demo_understanding(model, processor)
    demo_generation(model, processor)


if __name__ == "__main__":
    main()

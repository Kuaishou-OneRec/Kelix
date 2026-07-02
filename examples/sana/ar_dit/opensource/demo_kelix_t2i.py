#!/usr/bin/env python3
"""
Kelix demo 2: text-to-image generation (1024x1024).

Minimal demo — the heavy lifting (model loading, flow-matching sampling, VAE
decode) lives in `kelix_utils.py`. This script just shows the core flow:
  1. Load Kelix AR model + Kelix-DiT + DC-AE VAE.
  2. Write a prompt.
  3. Generate a 1024x1024 image and save it.

Usage:
    python examples/sana/ar_dit/opensource/demo_kelix_t2i.py
    python examples/sana/ar_dit/opensource/demo_kelix_t2i.py --prompt "Generate an image of a cute cat."
    python examples/sana/ar_dit/opensource/demo_kelix_t2i.py --output out.png

    # Override paths via env vars:
    KELIX_DIR=/path/to/release_sft DIT_DIR=/path/to/release_dit \
    VAE_DIR=/path/to/vae python examples/sana/ar_dit/opensource/demo_kelix_t2i.py
"""

import argparse
import os

# Make `kelix_utils` importable when this script is run directly
# (`python demo_kelix_t2i.py`) regardless of the current working directory.
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kelix_utils import (
    KELIX_DIR,
    DIT_DIR,
    VAE_DIR,
    load_ar_model,
    load_dit,
    load_processor,
    load_vae_model,
    generate_image,
)


def main():
    parser = argparse.ArgumentParser(description="Kelix text-to-image demo")
    parser.add_argument("--prompt", type=str, default="Generate an image of a cute cat.")
    parser.add_argument("--output", type=str, default="./kelix_t2i_demo.png")
    args = parser.parse_args()

    # 1. Load Kelix AR model + Kelix-DiT + DC-AE VAE.
    print(f"Loading AR model from {KELIX_DIR} ...")
    ar_model = load_ar_model()
    ar_processor = load_processor()

    print(f"Loading DiT from {DIT_DIR} ...")
    dit = load_dit()

    print(f"Loading VAE from {VAE_DIR} ...")
    vae = load_vae_model()

    # 2. Write a prompt -> 3. generate a 1024x1024 image.
    print(f"\nPrompt: {args.prompt}")
    img = generate_image(ar_model, ar_processor, dit, vae, args.prompt)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    img.save(args.output, quality=95)
    print(f"\nSaved: {args.output}  ({img.size})")


if __name__ == "__main__":
    main()

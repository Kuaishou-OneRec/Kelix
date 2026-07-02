#!/usr/bin/env python3
"""
Kelix demo 1: image understanding + image-token generation.

Minimal demo — the heavy lifting (model loading, chat template, generation)
lives in `kelix_utils.py`. This script just shows the core flow:
  1. Load the Kelix unified model.
  2. Image understanding: image + question -> text answer.
  3. Image-token generation: text prompt -> discrete visual tokens.

Usage:
    python examples/sana/ar_dit/opensource/demo_kelix.py
    KELIX_DIR=/path/to/release_sft python examples/sana/ar_dit/opensource/demo_kelix.py
"""

import torch

# Make `kelix_utils` importable when this script is run directly
# (`python demo_kelix.py`) regardless of the current working directory.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kelix_utils import (
    KELIX_DIR,
    load_ar_model,
    load_processor,
    make_test_image,
    chat,
    generate_image_tokens,
)


def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # 1. Load the Kelix unified model (Kelix-Tok + Kelix-LLM).
    print(f"Loading Kelix from {KELIX_DIR} ...")
    model = load_ar_model()
    processor = load_processor()
    model.eval()

    # 2. Image understanding: feed an image + question, get a text answer.
    print("\n" + "=" * 60)
    print("Task 1: Image Understanding")
    print("=" * 60)
    image = make_test_image()
    answer = chat(
        model, processor,
        messages=[
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What's in the image? Describe it briefly."},
            ]}
        ],
        max_new_tokens=256,
    )
    print(f"User : What's in the image?\nKelix: {answer}")

    # 3. Image-token generation: text prompt -> discrete visual tokens.
    print("\n" + "=" * 60)
    print("Task 2: Image-Token Generation")
    print("=" * 60)
    prompt = "Generate an image of a cute cat."
    output_ids, content, image_token_groups = generate_image_tokens(model, processor, prompt)
    print(f"User : {prompt}")
    print(f"Kelix (raw tokens): {content}")
    # Print the full output_ids tensor without PyTorch's "..." truncation.
    torch.set_printoptions(threshold=10**9, profile="default")
    print(f"output_ids (shape={tuple(output_ids.shape)}):\n{output_ids}")
    print(f"input_image_ids({[x.shape for x in image_token_groups]})=\n{image_token_groups}")


if __name__ == "__main__":
    main()

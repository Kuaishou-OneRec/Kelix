# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Sana AE Inference Script.

This script performs image reconstruction inference using a trained Sana AE model.
The condition is from KeyeImageTokenizer.

Usage:
    python recipes/inference_sana_ae.py \
        --model-dir /path/to/model \
        --vae-dir mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
        --image-tokenizer-dir /path/to/tokenizer \
        --input-dir /path/to/images \
        --output-dir /path/to/output
"""

import torch
import argparse
from transformers import AutoProcessor

from muse.models.base import Model
from muse.training.common import get_torch_dtype

# Import helper functions
from recipes.sana.utils import (
    load_vae,
    load_image_tokenizer,
    run_dit_reconstruction,
)


def get_argument_parser():
    parser = argparse.ArgumentParser(description="Sana AE Inference")

    # Model args
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the trained model directory (HuggingFace format)")
    
    # VAE args
    parser.add_argument("--vae-dir", type=str,
                        default="mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
                        help="Pretrained VAE model path")

    # Image Tokenizer args
    parser.add_argument("--image-tokenizer-dir", type=str, required=True,
                        help="Image tokenizer model path")
    
    parser.add_argument("--max-condition-length", type=int, default=324,
                        help="Maximum condition sequence length")

    parser.add_argument("--fusion-type", type=str, default="mean",
                        choices=["mean", "sum"],
                        help="Fusion type for image tokenizer: mean/sum")

    # Input/Output args
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input images")
    
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save output images")

    # Inference args
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Image size for inference")
    
    parser.add_argument("--cfg-scale", type=float, default=4.5,
                        help="CFG scale for sampling")
    
    parser.add_argument("--num-sampling-steps", type=int, default=20,
                        help="Number of Euler sampling steps")
    
    parser.add_argument("--flow-shift", type=float, default=3.0,
                        help="Flow shift parameter")
    
    parser.add_argument("--num-images", type=int, default=None,
                        help="Maximum number of images to process (default: all)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Device args
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Data type for inference")

    return parser


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    
    # Setup device and dtype
    device = torch.device(args.device)
    dtype = get_torch_dtype(args.dtype)
    
    print("=" * 60)
    print("Sana AE Inference")
    print("=" * 60)
    print(f"Model: {args.model_dir}")
    print(f"VAE: {args.vae_dir}")
    print(f"Image Tokenizer: {args.image_tokenizer_dir}")
    print(f"Device: {device}, Dtype: {dtype}")
    print("=" * 60)
    
    # Load DiT model
    print("Loading DiT model...")
    model = Model.from_pretrained(args.model_dir)
    model = model.to(device=device, dtype=dtype).eval()
    print(f"  Model loaded: {type(model).__name__}")
    
    # Load VAE
    print("Loading VAE...")
    vae = load_vae(args.vae_dir, device=device, dtype=dtype)
    
    # Load Image Tokenizer
    print("Loading Image Tokenizer...")
    image_tokenizer = load_image_tokenizer(
        args.image_tokenizer_dir, device=device, dtype=dtype,
        fusion_type=args.fusion_type
    )
    
    # Load processor for image preprocessing
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.image_tokenizer_dir, trust_remote_code=True)
    
    # Run inference
    run_dit_reconstruction(
        model=model,
        vae=vae,
        image_tokenizer=image_tokenizer,
        processor=processor,
        image_dir=args.input_dir,
        output_dir=args.output_dir,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        flow_shift=args.flow_shift,
        max_condition_length=args.max_condition_length,
        image_size=args.image_size,
        device=device,
        dtype=dtype,
        seed=args.seed,
        num_images=args.num_images,
    )
    
    print("=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

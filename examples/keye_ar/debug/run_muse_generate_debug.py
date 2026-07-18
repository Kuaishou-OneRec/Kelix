#!/usr/bin/env python3
"""Run Muse generate() with full debug to diagnose the row-15 degeneration bug.

This script:
  1. Loads the Muse model from release_sft
  2. Runs generate() with debug=True and MUSE_DEBUG_ATTN=1
  3. Prints detailed info about:
     - Prefill: input_pos, cache_pos, is_causal, logits
     - Each decode step: input_pos, cache_pos, is_causal, top-5 logits
     - Attention layer: q_len, k_len, is_causal, cache_pos (for layer 0)

The key hypothesis to verify:
  The attention layer uses `is_causal = (kv_cache is None) and ...`
  When kv_cache is set up (during generate), is_causal = False for BOTH
  prefill and decode. This means the PREFILL is non-causal — each prompt
  token sees all future tokens, contaminating the KV cache.

Usage:
  MUSE_DEBUG_ATTN=1 PYTHONPATH=. python3 examples/keye_ar/debug/run_muse_generate_debug.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

from muse.models.keye_ar import KeyeARModel
from muse.training.common import set_default_dtype


def load_muse_model(model_dir, device, dtype=torch.bfloat16):
    """Load the Muse (Muse-format) model."""
    print(f"[muse] Loading from {model_dir} ...")
    with set_default_dtype(dtype), torch.device(device):
        model = KeyeARModel.from_pretrained(model_dir).eval()
    model.config.qwen_config.output_last_hidden_states_only = False
    model.model.model.output_last_hidden_states_only = False
    model.set_token_decoder_with_teacher_forcing(False)
    model.requires_grad_(False)
    model = model.to(device=device, dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    print("[muse] Model loaded.")
    return model, processor


def build_prompt_inputs(processor, device, prompt):
    """Build model inputs from a text prompt (no image)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    from keye_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, *_ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=False, truncation=False, return_tensors="pt",
    ).to(device)
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",
                        default="/mmu_mllm_hdd_2/lingzhixin/output/release/muse_eval/release_sft")
    parser.add_argument("--prompt", default="Generate an image of a cat.")
    parser.add_argument("--max-new-tokens", type=int, default=450,
                        help="Max groups to generate (18x18 image needs ~434)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Decoding temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=1, help="Top-K (1 = greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-P (1.0 = no filter)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = args.device

    # ==============================================
    # Load model
    # ==============================================
    print(f"\n{'#' * 70}")
    print(f"# Load Muse model and run generate() with full debug")
    print(f"{'#' * 70}")

    model, processor = load_muse_model(args.model_dir, device, dtype)
    inputs = build_prompt_inputs(processor, device, args.prompt)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.size(1)
    print(f"  Prompt: {args.prompt}")
    print(f"  input_ids shape: {tuple(input_ids.shape)}")

    qcfg = model.config.qwen_config
    n_tokens = qcfg.n_q_tokens + 1  # 9
    print(f"  config: n_q={qcfg.n_q_tokens}, n_tok={n_tokens}, "
          f"img={qcfg.image_token_id}, qeos={qcfg.q_eos_token}, "
          f"eos={qcfg.eos_token_id}, vs={qcfg.vision_start_token_id}, "
          f"ve={qcfg.vision_end_token_id}")

    # ==============================================
    # Run generate() with debug=True
    # ==============================================
    print(f"\n{'#' * 70}")
    print(f"# Running generate() with debug=True, max_new_tokens={args.max_new_tokens}")
    print(f"# MUSE_DEBUG_ATTN={os.environ.get('MUSE_DEBUG_ATTN', '0')}")
    print(f"{'#' * 70}\n")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    is_greedy = args.temperature == 0 or args.top_k == 1
    print(f"  Decoding mode: {'greedy' if is_greedy else 'sampling'}")
    print(f"  temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, seed={args.seed}")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            debug=True,
        )

    # ==============================================
    # Summary
    # ==============================================
    print(f"\n{'#' * 70}")
    print(f"# Generation Summary")
    print(f"{'#' * 70}")
    print(f"  output_ids shape: {tuple(output_ids.shape)}")
    generated = output_ids[0, prompt_len:, :]  # (gen_len, 9)
    gen_first_col = generated[:, 0].tolist()
    print(f"  generated length: {len(gen_first_col)} groups")
    print(f"  first_col (first 20): {gen_first_col[:20]}")
    print(f"  first_col (last 20):  {gen_first_col[-20:]}")
    print(f"  vision_start count: {gen_first_col.count(qcfg.vision_start_token_id)}")
    print(f"  vision_end count:   {gen_first_col.count(qcfg.vision_end_token_id)}")
    print(f"  eos count:          {gen_first_col.count(qcfg.eos_token_id)}")
    print(f"  <|mm_pos_start|> (151682) count: {gen_first_col.count(151682)}")

    # Check if the model degenerated (produced <|mm_pos_start|> where it shouldn't)
    mm_pos_start_id = 151682
    mm_pos_positions = [i for i, t in enumerate(gen_first_col) if t == mm_pos_start_id]
    print(f"\n  <|mm_pos_start|> positions: {mm_pos_positions}")
    if len(mm_pos_positions) > 18:
        print(f"  *** DEGENERATION DETECTED: {len(mm_pos_positions)} <|mm_pos_start|> markers "
              f"(expected 18 for an 18x18 grid)")
        # Find the first degenerate marker (where the model loops)
        for i, pos in enumerate(mm_pos_positions):
            if i > 0 and pos == mm_pos_positions[i-1] + 1:
                print(f"  First degenerate loop at gen idx {pos}: "
                      f"<|mm_pos_start|> right after previous <|mm_pos_start|>")
                break


if __name__ == "__main__":
    main()

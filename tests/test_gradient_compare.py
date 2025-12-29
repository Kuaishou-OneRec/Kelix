"""
Gradient Comparison Test Script for msy_master_2/muse
=======================================================
Purpose: Compare backward gradients between two repositories.

This script:
1. Loads model from checkpoint
2. Uses real video input via Processor (same as test_keye_vl_video_muse_only.py)
3. Uses fixed target text for loss computation
4. Runs forward + backward pass
5. Saves gradients to file for comparison

Usage:
    cd msy_master_2/muse
    python tests/test_gradient_compare.py --model-dir /path/to/model --output-path gradients_msy.pt
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["nosp"] = 'true'  # Disable sequence parallel

import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
import tqdm
from transformers import AutoProcessor

# === Muse imports ===
from muse.models import get_model_class
from muse.config import load_config
from muse.training.common import set_default_dtype

# === Import Processor utils ===
try:
    from tests.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info
except ImportError:
    sys.path.append(os.getcwd())
    from tests.models.tokenizer_end2end_mt_1drope_video.keye_vl_utils import process_vision_info

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

# Default values (can be overridden by command line args)
DEFAULT_MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"
DEFAULT_OUTPUT_PATH = "/tmp/gradients_msy_master_2.pt"

# Real video input path
DEFAULT_VIDEO_PATH = "/llm_reco/maosiyang/23b77760a4304e9092eb3b45b7bf8050.mp4"

# Fixed target text for loss computation (must be same across both repos!)
TARGET_TEXT = "一个男人站在镜头前"

# Fixed seed for reproducibility
SEED = 42

# Device and dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_real_inputs(model_dir, video_path, target_text, device, dtype):
    """
    Create real inputs using Processor, same as test_keye_vl_video_muse_only.py.
    
    Args:
        model_dir: Path to model directory (for loading processor)
        video_path: Path to input video
        target_text: Target text for loss computation
        device: torch device
        dtype: torch dtype
    
    Returns:
        dict: Dictionary containing model inputs and labels
    """
    # Load processor
    logger.info(f"Loading Processor from {model_dir}...")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    
    # Construct input message with video
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
            ],
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    logger.info(f"Prompt text: {repr(text[:100])}...")
    
    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)
    logger.info(f"process_vision_info: images={len(image_inputs) if image_inputs else 0}, videos={len(video_inputs) if video_inputs else 0}")
    
    # Run processor to get model inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    ).to(device)
    
    # Remove num_frames if present (not needed by model)
    inputs.pop("num_frames", None)
    
    # Create labels from target text
    # We need to create a proper label sequence that matches the input format
    # The target should come after the video/prompt tokens
    target_ids = processor.tokenizer.encode(target_text, add_special_tokens=False)
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    
    # Create labels: copy input_ids and append target
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    # Labels for autoregressive: shift by 1
    # We compute loss on the target tokens only
    # Append target_ids to input_ids for the full sequence
    full_input_ids = torch.cat([input_ids, target_ids.unsqueeze(0)], dim=1)
    
    # Labels: same as full_input_ids
    labels = full_input_ids.clone()
    
    # Loss mask: only compute loss on target tokens (after original input)
    loss_mask = torch.zeros_like(full_input_ids)
    loss_mask[0, seq_len:] = 1  # Only loss on target portion
    
    # Update attention mask
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    full_attention_mask = torch.cat([
        attention_mask, 
        torch.ones(1, len(target_ids), dtype=attention_mask.dtype, device=device)
    ], dim=1)
    
    return {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "pixel_values_videos": inputs.get("pixel_values_videos"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "labels": labels,
        "loss_mask": loss_mask,
        "target_text": target_text,
        "original_seq_len": seq_len,
    }


def compute_loss(logits, labels, loss_mask, ignore_index=-100):
    """
    Compute cross-entropy loss with masking.
    
    Args:
        logits: [B, S, V] model output logits
        labels: [B, S] target token IDs
        loss_mask: [B, S] mask indicating which positions to compute loss
        ignore_index: index to ignore in loss computation
    
    Returns:
        loss: scalar tensor
    """
    # Shift for autoregressive prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()
    
    # Apply mask to labels
    masked_labels = shift_labels.clone()
    masked_labels[shift_mask == 0] = ignore_index
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    masked_labels = masked_labels.view(-1)
    
    # Compute loss
    loss = F.cross_entropy(shift_logits, masked_labels, ignore_index=ignore_index)
    
    return loss


def collect_gradients(model) -> dict:
    """
    Collect gradients from all model parameters.
    
    Returns:
        dict: Dictionary mapping parameter names to gradient statistics
    """
    grad_dict = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().float()  # Convert to float32 for statistics
            grad_dict[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "max": grad.max().item(),
                "min": grad.min().item(),
                "norm": grad.norm().item(),
                "abs_mean": grad.abs().mean().item(),
                "shape": list(grad.shape),
                "numel": grad.numel(),
                "grad_tensor": param.grad.detach().cpu().clone(),  # Store full gradient
            }
        else:
            grad_dict[name] = {
                "mean": None,
                "std": None,
                "max": None,
                "min": None,
                "norm": None,
                "abs_mean": None,
                "shape": list(param.shape),
                "numel": param.numel(),
                "grad_tensor": None,
                "no_grad": True,
            }
    
    return grad_dict


def main():
    parser = argparse.ArgumentParser(description="Gradient comparison test for msy_master_2/muse")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Path to model directory")
    parser.add_argument("--video-path", type=str, default=DEFAULT_VIDEO_PATH,
                        help="Path to input video")
    parser.add_argument("--target-text", type=str, default=TARGET_TEXT,
                        help="Target text for loss computation")
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Path to save gradients")
    parser.add_argument("--save-full-grads", action="store_true",
                        help="Save full gradient tensors (may be large)")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Gradient Comparison Test - msy_master_2/muse")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}, Dtype: {DTYPE}")
    logger.info(f"Model Dir: {args.model_dir}")
    logger.info(f"Video Path: {args.video_path}")
    logger.info(f"Target Text: {args.target_text}")
    logger.info(f"Output Path: {args.output_path}")
    logger.info(f"Seed: {SEED}")
    
    # Set global seed
    set_seed(SEED)
    
    # --- 1. Load config ---
    config_path = Path(args.model_dir) / "muse_config.json"
    logger.info(f"Loading config from {config_path}...")
    model_config = load_config(config_path)
    
    # --- 2. Create model ---
    model_class_name = model_config.model_class
    logger.info(f"Creating model: {model_class_name}...")
    model_cls = get_model_class(model_class_name)
    
    with set_default_dtype(DTYPE):
        model = model_cls(model_config)
    
    # --- 3. Load weights ---
    logger.info(f"Loading weights from {args.model_dir}...")
    sd = {}
    safetensor_files = [f for f in os.listdir(args.model_dir) if f.endswith(".safetensors")]
    for f in tqdm.tqdm(safetensor_files, desc="Loading safetensors"):
        sd.update(load_file(os.path.join(args.model_dir, f)))
    
    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")
    
    model = model.to(DEVICE).to(DTYPE)
    
    # Important: Set to training mode for gradient computation
    model.train()
    
    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("Model loaded successfully!")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # --- 4. Create real inputs using Processor ---
    logger.info("Creating real inputs from video...")
    inputs = create_real_inputs(args.model_dir, args.video_path, args.target_text, DEVICE, DTYPE)
    
    logger.info(f"  input_ids shape: {inputs['input_ids'].shape}")
    if inputs.get('pixel_values_videos') is not None:
        logger.info(f"  pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
    if inputs.get('video_grid_thw') is not None:
        logger.info(f"  video_grid_thw: {inputs['video_grid_thw'].tolist()}")
    logger.info(f"  target_text: {inputs['target_text']}")
    logger.info(f"  original_seq_len: {inputs['original_seq_len']}")
    logger.info(f"  labels shape: {inputs['labels'].shape}")
    logger.info(f"  loss_mask sum: {inputs['loss_mask'].sum().item()}")
    
    # --- 5. Forward pass ---
    logger.info("Running forward pass...")
    
    # Prepare model inputs (exclude loss_mask, labels, and metadata from model call)
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    
    # Add video inputs if present
    if inputs.get("pixel_values_videos") is not None:
        model_inputs["pixel_values_videos"] = inputs["pixel_values_videos"]
    if inputs.get("video_grid_thw") is not None:
        model_inputs["video_grid_thw"] = inputs["video_grid_thw"]
    
    # Add image inputs if present
    if inputs.get("pixel_values") is not None:
        model_inputs["pixel_values"] = inputs["pixel_values"]
    if inputs.get("image_grid_thw") is not None:
        model_inputs["image_grid_thw"] = inputs["image_grid_thw"]
    
    outputs = model(**model_inputs)
    
    # Extract logits
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        # Also get auxiliary losses if available
        codebook_loss = outputs.get("codebook_loss", [])
        commitment_loss = outputs.get("commitment_loss", [])
    else:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        codebook_loss = []
        commitment_loss = []
    
    logger.info(f"  Logits shape: {logits.shape}")
    logger.info(f"  Logits dtype: {logits.dtype}")
    
    # --- 6. Compute loss ---
    logger.info("Computing loss...")
    
    lm_loss = compute_loss(logits, inputs["labels"], inputs["loss_mask"])
    
    # Add auxiliary losses if available
    total_loss = lm_loss
    if codebook_loss:
        cb_loss = sum(codebook_loss) / len(codebook_loss) if isinstance(codebook_loss, list) else codebook_loss
        total_loss = total_loss + cb_loss
        logger.info(f"  Codebook loss: {cb_loss.item() if hasattr(cb_loss, 'item') else cb_loss}")
    if commitment_loss:
        cm_loss = sum(commitment_loss) / len(commitment_loss) if isinstance(commitment_loss, list) else commitment_loss
        total_loss = total_loss + 0.25 * cm_loss
        logger.info(f"  Commitment loss: {cm_loss.item() if hasattr(cm_loss, 'item') else cm_loss}")
    
    logger.info(f"  LM loss: {lm_loss.item()}")
    logger.info(f"  Total loss: {total_loss.item()}")
    
    # --- 7. Backward pass ---
    logger.info("Running backward pass...")
    
    model.zero_grad()
    total_loss.backward()
    
    logger.info("Backward pass completed!")
    
    # --- 8. Collect gradients ---
    logger.info("Collecting gradients...")
    
    grad_dict = collect_gradients(model)
    
    # Remove full gradient tensors if not requested
    if not args.save_full_grads:
        for name in grad_dict:
            if "grad_tensor" in grad_dict[name]:
                del grad_dict[name]["grad_tensor"]
    
    # Count parameters with gradients
    params_with_grad = sum(1 for v in grad_dict.values() if v.get("mean") is not None)
    params_without_grad = sum(1 for v in grad_dict.values() if v.get("no_grad", False))
    
    logger.info(f"  Parameters with gradients: {params_with_grad}")
    logger.info(f"  Parameters without gradients: {params_without_grad}")
    
    # Print parameters without gradients
    if params_without_grad > 0:
        logger.info("  Parameters without gradients:")
        for name, info in grad_dict.items():
            if info.get("no_grad", False):
                logger.info(f"    - {name} (shape: {info['shape']})")
    
    # --- 9. Save gradients ---
    logger.info(f"Saving gradients to {args.output_path}...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Compute input stats
    input_stats = {
        "input_ids_sum": inputs["input_ids"].sum().item(),
        "input_ids_shape": list(inputs["input_ids"].shape),
    }
    if inputs.get("pixel_values_videos") is not None:
        input_stats["pixel_values_videos_mean"] = inputs["pixel_values_videos"].float().mean().item()
        input_stats["pixel_values_videos_std"] = inputs["pixel_values_videos"].float().std().item()
        input_stats["pixel_values_videos_shape"] = list(inputs["pixel_values_videos"].shape)
    if inputs.get("video_grid_thw") is not None:
        input_stats["video_grid_thw"] = inputs["video_grid_thw"].tolist()
    
    # Save metadata along with gradients
    save_dict = {
        "gradients": grad_dict,
        "metadata": {
            "model_class": model_class_name,
            "model_dir": args.model_dir,
            "video_path": args.video_path,
            "target_text": args.target_text,
            "seed": SEED,
            "dtype": str(DTYPE),
            "device": DEVICE,
            "seq_len": inputs["input_ids"].shape[1],
            "original_seq_len": inputs["original_seq_len"],
            "lm_loss": lm_loss.item(),
            "total_loss": total_loss.item(),
            "framework": "msy_master_2/muse",
        },
        "input_stats": input_stats,
    }
    
    torch.save(save_dict, args.output_path)
    logger.info("Gradients saved successfully!")
    
    # --- 10. Print summary ---
    logger.info("\n" + "=" * 60)
    logger.info("Gradient Summary (Top 10 by norm)")
    logger.info("=" * 60)
    
    # Sort by gradient norm
    sorted_grads = sorted(
        [(k, v) for k, v in grad_dict.items() if v.get("norm") is not None],
        key=lambda x: x[1]["norm"],
        reverse=True
    )[:10]
    
    for name, stats in sorted_grads:
        logger.info(f"{name}:")
        logger.info(f"  norm={stats['norm']:.6e}, mean={stats['mean']:.6e}, std={stats['std']:.6e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


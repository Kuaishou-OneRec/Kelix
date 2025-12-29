"""
Gradient Comparison Test Script for msy_master_2/muse
=======================================================
Purpose: Compare backward gradients between two repositories.

This script:
1. Loads model from checkpoint
2. Constructs deterministic fake inputs
3. Runs forward pass
4. Computes loss with fake target
5. Runs backward pass
6. Saves gradients to file for comparison

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

# === Muse imports ===
from muse.models import get_model_class
from muse.config import load_config
from muse.training.common import set_default_dtype

logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

# Default values (can be overridden by command line args)
DEFAULT_MODEL_DIR = "/llm_reco_ssd/maosiyang/models/muse/keye_tokenizer_end2end_image_for_stage_2_video"
DEFAULT_OUTPUT_PATH = "/tmp/gradients_msy_master_2.pt"

# Fixed seed for reproducibility
SEED = 42

# Device and dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Fake input configuration
FAKE_SEQ_LEN = 512
FAKE_NUM_VIDEO_PATCHES = 256  # 4 frames * 8*8 spatial
FAKE_VIDEO_GRID_THW = [4, 8, 8]  # 4 frames, 8x8 patches per frame
FAKE_PATCH_SIZE = 14


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_fake_inputs(model_config, device, dtype):
    """
    Create deterministic fake inputs for gradient testing.
    
    Returns:
        dict: Dictionary containing all model inputs
    """
    set_seed(SEED)
    
    # Get token IDs from config
    video_token_id = getattr(model_config, 'video_token_id', 151656)
    pad_token_id = 0
    
    # Create input_ids with video tokens
    input_ids = torch.ones(1, FAKE_SEQ_LEN, dtype=torch.long, device=device) * pad_token_id
    
    # Insert video tokens at specific positions
    video_start_pos = 10
    video_end_pos = video_start_pos + FAKE_NUM_VIDEO_PATCHES
    input_ids[0, video_start_pos:video_end_pos] = video_token_id
    
    # Fill remaining positions with some text tokens
    set_seed(SEED + 1)
    text_token_range = (1000, 10000)
    text_positions = torch.cat([
        torch.arange(1, video_start_pos),
        torch.arange(video_end_pos, FAKE_SEQ_LEN)
    ])
    for pos in text_positions:
        input_ids[0, pos] = torch.randint(text_token_range[0], text_token_range[1], (1,)).item()
    
    # Create attention mask (all ones for simplicity)
    attention_mask = torch.ones(1, FAKE_SEQ_LEN, dtype=torch.long, device=device)
    
    # Create fake pixel values for video: [num_patches, C, H, W]
    set_seed(SEED + 2)
    pixel_values_videos = torch.randn(
        FAKE_NUM_VIDEO_PATCHES, 3, FAKE_PATCH_SIZE, FAKE_PATCH_SIZE,
        dtype=dtype, device=device
    )
    
    # Create video grid THW
    video_grid_thw = torch.tensor([FAKE_VIDEO_GRID_THW], dtype=torch.long, device=device)
    
    # Create labels (same as input_ids for autoregressive)
    labels = input_ids.clone()
    
    # Create loss mask (only compute loss on non-video tokens after video region)
    loss_mask = torch.zeros(1, FAKE_SEQ_LEN, dtype=torch.long, device=device)
    loss_mask[0, video_end_pos:] = 1  # Only compute loss after video tokens
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "labels": labels,
        "loss_mask": loss_mask,
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
    
    # --- 4. Create fake inputs ---
    logger.info("Creating fake inputs...")
    inputs = create_fake_inputs(model_config, DEVICE, DTYPE)
    
    logger.info(f"  input_ids shape: {inputs['input_ids'].shape}")
    logger.info(f"  pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
    logger.info(f"  video_grid_thw: {inputs['video_grid_thw'].tolist()}")
    logger.info(f"  labels shape: {inputs['labels'].shape}")
    logger.info(f"  loss_mask sum: {inputs['loss_mask'].sum().item()}")
    
    # --- 5. Forward pass ---
    logger.info("Running forward pass...")
    
    # Prepare model inputs (exclude loss_mask and labels from model call)
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "pixel_values_videos": inputs["pixel_values_videos"],
        "video_grid_thw": inputs["video_grid_thw"],
    }
    
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
    
    # --- 9. Save gradients ---
    logger.info(f"Saving gradients to {args.output_path}...")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save metadata along with gradients
    save_dict = {
        "gradients": grad_dict,
        "metadata": {
            "model_class": model_class_name,
            "model_dir": args.model_dir,
            "seed": SEED,
            "dtype": str(DTYPE),
            "device": DEVICE,
            "seq_len": FAKE_SEQ_LEN,
            "num_video_patches": FAKE_NUM_VIDEO_PATCHES,
            "video_grid_thw": FAKE_VIDEO_GRID_THW,
            "lm_loss": lm_loss.item(),
            "total_loss": total_loss.item(),
            "framework": "msy_master_2/muse",
        },
        "input_stats": {
            "input_ids_sum": inputs["input_ids"].sum().item(),
            "pixel_values_mean": inputs["pixel_values_videos"].mean().item(),
            "pixel_values_std": inputs["pixel_values_videos"].std().item(),
        }
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


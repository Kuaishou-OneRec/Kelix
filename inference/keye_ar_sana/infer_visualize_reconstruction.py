"""
Inference demo: visualize DiT reconstructions using Keye AR processor

This script mirrors the visualization pipeline in
`recipes/sana/train_sana_ar_dit.py` and is intended as a ready-to-run demo for
inference/visualization using the parquet specified in
`run_ar_dit_multiscale_cross_1280tokens_attn_v1.sh`.

Usage example:
    python inference/keye_ar_sana/infer_visualize_reconstruction.py \
        --model-dir /llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px/ \
        --vae-dir /llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/ \
        --keye-ar-dir /mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.9/v2_stage3_1e-4_max1280/./step23000/global_step23000/muse_converted \
        --dataset-config examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im_multiscale.json \
        --parquet-path /mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet \
        --output-dir /tmp/vis_demo --num-images 8

This demo imports and reuses helper functions from
`recipes/sana/train_sana_ar_dit.py` (e.g., `load_vae`, `load_keye_ar`,
`visualize_reconstruction`) to ensure parity with the training pipeline.
"""

import argparse
import os
import json
import torch
import torch.distributed as dist
from pathlib import Path

# Reuse helpers from the training recipe
from recipes.sana import train_sana_ar_dit as train_rec
from muse.config import load_config
from muse.models import get_model_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing pretrained model or checkpoint")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Optional model config JSON path (if not using --model-dir/config.json)")
    parser.add_argument("--vae-dir", type=str, required=True,
                        help="VAE directory")
    parser.add_argument("--keye-ar-dir", type=str, required=True,
                        help="Keye AR processor/model directory")
    parser.add_argument("--dataset-config", type=str,
                        default="examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im_multiscale.json",
                        help="Dataset config JSON used to build Chat2ImageDataset")
    parser.add_argument("--parquet-path", type=str,
                        default="/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet",
                        help="Parquet file for visualization samples")
    parser.add_argument("--output-dir", type=str, default="./vis_output",
                        help="Directory to save visualization outputs")
    parser.add_argument("--num-images", type=int, default=12,
                        help="Max number of images to visualize")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda|cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model compute dtype to use")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--max-condition-length", type=int, default=2560)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Directory to save generated DiT JPEGs and messages JSON")

    # Distributed options: Chat2ImageDataset/Token2ImageDataset expect rank/world_size
    parser.add_argument("--initialize-dist", action="store_true",
                        help="Initialize a local single-process distributed group for dataset compatibility")
    parser.add_argument("--rank", type=int, default=0,
                        help="Distributed rank to set in dataset config (default: 0)")
    parser.add_argument("--world-size", type=int, default=1,
                        help="Distributed world_size to set in dataset config (default: 1)")
    return parser.parse_args()


def setup_distributed_environment(rank: int = 0, world_size: int = 1) -> bool:
    """
    Initialize distributed environment for single-process inference runs using the
    same approach as our tests: TCP init on localhost (gloo backend).

    Returns:
        True if distributed was initialized successfully, otherwise False.
    """
    print(f"going to init: {rank}/{world_size}")
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:29500',
        rank=rank,
        world_size=world_size,
    )
    print("Initialized local TCP-based distributed process group (127.0.0.1:29500)")
    return True



def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = train_rec.get_torch_dtype(args.dtype) if hasattr(train_rec, 'get_torch_dtype') else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally initialize a local single-process distributed group for dataset compatibility
    setup_distributed_environment(args.rank, args.world_size)

    # 1) Load model config and instantiate model for visualization
    if args.model_config:
        model_config = load_config(args.model_config)
    else:
        # Expect config.json in model dir
        cfg_path = Path(args.model_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Model config not found at {cfg_path}. Provide --model-config if needed.")
        model_config = load_config(cfg_path)

    model_class_name = model_config.model_class
    model_cls = get_model_class(model_class_name)

    print(f"Creating visualization model: {model_class_name}")
    with train_rec.set_default_dtype(args.dtype), torch.device("cpu"):
        model_for_vis = model_cls(model_config)

    # 2) Try to load checkpoint/state dict from model_dir
    try:
        if hasattr(train_rec, 'load_hf_checkpoint'):
            print(f"Loading checkpoint from {args.model_dir}")
            sd = train_rec.load_hf_checkpoint(args.model_dir)
            # Load state into model_for_vis (best-effort)
            try:
                model_for_vis.load_state_dict(sd, strict=False)
                print("Model weights loaded (strict=False)")
            except Exception as e:
                print(f"Warning: failed to load state_dict with strict=False: {e}")
                # Try key 'model' or 'app' wrappers
                if isinstance(sd, dict) and 'model' in sd:
                    try:
                        model_for_vis.load_state_dict(sd['model'], strict=False)
                        print("Model weights loaded from sd['model']")
                    except Exception:
                        print("Failed to load model weights from sd['model']")
        else:
            print("load_hf_checkpoint not available in train_rec; skipping checkpoint load")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    model_for_vis.to(device)
    model_for_vis.eval()

    # 3) Load VAE and Keye AR tokenizer/processor
    print("Loading VAE...")
    vae = train_rec.load_vae(args.vae_dir, device=device, dtype=dtype)

    print("Loading Keye AR tokenizer/processor...")

    # # Initialize a local single-process distributed group to ensure that
    # # training helpers which call `torch.distributed.get_rank()` behave correctly.
    # setup_distributed_environment(args.rank, args.world_size)

    image_tokenizer = train_rec.load_keye_ar(args.keye_ar_dir, device=device, dtype=args.dtype)
    # Ensure tokenizer/model is on the intended device (Triton kernels expect CUDA tensors)
    
    image_tokenizer = image_tokenizer.to(device)

    # 4) Build dataset using provided dataset config (for processing helpers)
    with open(args.dataset_config, encoding='utf-8') as f:
        dataset_cfg = json.load(f)

    # Ensure processor_path is set to Keye AR if not present
    if not dataset_cfg.get('processor_path'):
        dataset_cfg['processor_path'] = args.keye_ar_dir

    dataset_cfg['image_size'] = args.image_size
    dataset_cfg['max_condition_length'] = args.max_condition_length
    # Pass rank/world_size so datasets expecting distributed info work in single-process mode
    dataset_cfg['rank'] = args.rank
    dataset_cfg['world_size'] = args.world_size

    print(f"Building Chat2ImageDataset for visualization with config: {dataset_cfg}")
    dataset = train_rec.Chat2ImageDataset(**dataset_cfg)

    # 5) Run DiT sampling pipeline *locally* and save results (DiT JPEGs + messages JSON)
    print("Running DiT sampling and saving results...")
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    import time

    try:
        with torch.no_grad():
            # Load samples / preprocess
            loaded = train_rec.VisReconstructionLoader()(
                args.parquet_path,
                dataset,
                args.image_size,
                device,
                dtype,
                args.num_images,
                None,
                vae,
            )

            # Tokenize images to condition embeddings
            cond_embeds, cond_mask = train_rec.tokenize_images(
                tokenizer=image_tokenizer,
                pixel_values=loaded.pixel_values.to(device=device),
                image_grid_thw=loaded.image_grid_thw.to(device=device),
                batch_size=loaded.batch_size,
                max_condition_length=args.max_condition_length,
                input_ids=loaded.input_ids.to(device=device),
            )

            # Prepare unconditional embeddings for CFG
            null_embed = model_for_vis.y_embedder.y_embedding
            seq_len = min(null_embed.shape[0], args.max_condition_length)
            uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(loaded.batch_size, -1, -1)
            if seq_len < args.max_condition_length:
                padding = torch.zeros(
                    loaded.batch_size, args.max_condition_length - seq_len, uncond_embeds.shape[-1],
                    device=device, dtype=dtype
                )
                uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
            uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
            uncond_mask = torch.zeros(loaded.batch_size, args.max_condition_length, device=device)
            uncond_mask[:, :seq_len] = 1
            uncond_mask = uncond_mask[:, None, None, :]

            # Create scheduler and sample
            scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
            scheduler.set_timesteps(args.num_sampling_steps, device=device)

            generator = torch.Generator(device=device).manual_seed(args.seed)
            dit_latents = torch.randn(
                (loaded.batch_size, loaded.latent_channels, loaded.latent_size, loaded.latent_size),
                generator=generator,
                device=device,
                dtype=dtype,
            )

            cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
            mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)

            for t in scheduler.timesteps:
                latent_input = torch.cat([dit_latents] * 2)
                timestep = t.expand(latent_input.shape[0])
                noise_pred = model_for_vis.forward_with_dpmsolver(latent_input, timestep, cond_embeds_cfg, mask=mask_cfg)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.cfg_scale * (noise_cond - noise_uncond)
                dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]

            # Decode DiT latents
            dit_recon_latents = dit_latents / vae.config.scaling_factor
            dit_recon_images = vae.decode(dit_recon_latents).sample
            dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)

            # Save DiT JPEGs and messages JSON
            results_step_dir = os.path.join(args.results_dir, "step_0")
            os.makedirs(results_step_dir, exist_ok=True)

            dit_np = dit_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
            messages = list(getattr(loaded, 'texts', []))
            mapping = {}
            for i in range(len(dit_np)):
                img = Image.fromarray((dit_np[i] * 255).round().astype("uint8"))
                img_fname = f"dit_{i}.jpg"
                img.save(os.path.join(results_step_dir, img_fname), quality=95)
                mapping[img_fname] = messages[i] if i < len(messages) else ""

            # Save messages mapping
            with open(os.path.join(results_step_dir, "messages.json"), 'w', encoding='utf-8') as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(dit_np)} DiT images and messages to: {results_step_dir}")
    except Exception as e:
        print(f"Error running DiT sampling: {e}")
        raise


if __name__ == "__main__":
    main()

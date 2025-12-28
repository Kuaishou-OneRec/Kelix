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
For DCP checkpoint:
    python inference/keye_ar_sana/infer_visualize_reconstruction.py \
        --model-dir /mmu_mllm_hdd_2/lingzhixin/output/MuseV2/sana/ar_dit/exp11_run_ar_dit_multiscale_1280tokens_attnrope_128u \
        --dcp-source-dir /llm_reco_ssd/zhouyang12/models/muse/Sana_1600M_1024px/ \
        --dcp-tag global_step8000 \
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
import easydict
import torch.distributed as dist
import pickle  # Add pickle import
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoProcessor

# Import DCP to torch converter
from muse.tools.dcp2torch import convert as dcp_to_torch_convert

# Reuse helpers from the training recipe
from recipes.sana import train_sana_ar_dit as train_rec
from muse.config import load_config
from muse.models import get_model_class
from muse.models.keye_ar import KeyeARModel
from muse.utils.common import parse_config_overrides
from muse.data.datasets.image import GenEvalInferenceDataset
from muse.training.parallel import (
    initialize_model_parallel
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing pretrained model or checkpoint")
    parser.add_argument("--vae-dir", type=str, required=True,
                        help="Directory containing pretrained VAE")
    parser.add_argument("--keye-ar-dir", type=str, required=True,
                        help="Directory containing pretrained Keye AR processor")
    parser.add_argument("--dataset-config", type=str, required=True,
                        help="Path to dataset configuration JSON")
    parser.add_argument("--parquet-path", type=str, required=True,
                        help="Path to parquet file containing images")
    parser.add_argument("--output-dir", type=str, default="/tmp/vis_demo",
                        help="Directory to save visualization results")
    parser.add_argument("--num-images", type=int, default=8,
                        help="Number of images to visualize (default: 8)")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="Input image size (default: 1024)")
    parser.add_argument("--max-condition-length", type=int, default=1280,
                        help="Maximum length of condition embeddings (default: 1280)")
    parser.add_argument("--teacher-forcing", action="store_true",
                        help="Use teacher forcing for AR model (default: False)")
    parser.add_argument("--dcp-source-dir", type=str, default=None,
                        help="Source directory for DCP checkpoint conversion")
    parser.add_argument("--dcp-tag", type=str, default=None,
                        help="Tag for DCP checkpoint conversion")
    parser.add_argument("--gen_eval_csv_path", type=str, default="/llm_reco/lingzhixin/recovlm_data/generation_data/GenEval.tsv",
                        help="Path to GenEval.tsv file")
    parser.add_argument("--infer_repeats", type=int, default=4,
                        help="Number of times to repeat inference for each sample")
    return parser.parse_args()


def setup_distributed_environment() -> bool:
    """
    Initialize distributed environment for single-process inference runs using the
    same approach as our tests: TCP init on localhost (gloo backend).

    Returns:
        True if distributed was initialized successfully, otherwise False.
    """

    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    import datetime
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=3600)
    )
    initialize_model_parallel()
    return True



def get_model_embedding_and_tokens(
        model: KeyeARModel,
        teacher_forcing: bool = False,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
    if teacher_forcing:
        outputs = model(**kwargs)
        embeddings = outputs # .last_hidden_state  # [B, seq_len, embed_dim]
        return embeddings, embeddings
    else:
        if "input_pos" in kwargs:
            del kwargs["input_pos"]
        if "pixel_values" in kwargs:
            del kwargs["pixel_values"]
            del kwargs["image_grid_thw"]
        if "cu_seqlens" in kwargs:
            del kwargs["cu_seqlens"]

        model.set_output_hidden_states([len(model.model.model.layers)])
        tokens, embeddings = model.generate(
            input_ids=input_ids,
            max_new_tokens=kwargs.get("max_length", 1280),
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        # shape: [B, seq_len, embed_dim]
        embeddings = embeddings.hidden_states[0]
        return embeddings, tokens.sequences



def tokenize_images(
        ar_model: KeyeARModel,
        batch_size: int,
        max_condition_length: int,
        input_ids: torch.Tensor,
        teacher_forcing: bool = False,
        ar_processor: Optional[AutoProcessor] = None,
        **kwargs
    ):
    """
    Tokenize images using the AR model and processor.

    Args:
        ar_model: Keye AR model.
        batch_size: Batch size.
        max_condition_length: Maximum length of condition embeddings.
        input_ids: Input IDs for the AR model.
        teacher_forcing: Whether to use teacher forcing.
        ar_processor: AR processor.
        **kwargs: Additional arguments.

    Returns:
        Tuple of (condition embeddings, condition mask).
    """
    # Get embeddings from the AR model
    embeddings, tokens = get_model_embedding_and_tokens(
        model=ar_model,
        teacher_forcing=teacher_forcing,
        input_ids=input_ids,
        max_length=max_condition_length,
        **kwargs
    )

    # Process embeddings to match expected shape
    seq_len = min(embeddings.shape[1], max_condition_length)
    processed_embeddings = embeddings[:, :seq_len, :]

    # Create mask for embeddings
    mask = torch.zeros(batch_size, max_condition_length, device=embeddings.device)
    mask[:, :seq_len] = 1
    mask = mask[:, None, None, :]

    # Pad embeddings if necessary
    if seq_len < max_condition_length:
        padding = torch.zeros(
            batch_size, max_condition_length - seq_len, processed_embeddings.shape[-1],
            device=processed_embeddings.device, dtype=processed_embeddings.dtype
        )
        processed_embeddings = torch.cat([processed_embeddings, padding], dim=1)

    return processed_embeddings, mask


def main():
    args = parse_args()

    # Set up distributed environment
    setup_distributed_environment()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load VAE
    print("Loading VAE...")
    vae = train_rec.load_vae(args.vae_dir)
    vae.eval()

    # 2) Load Keye AR model and processor
    print("Loading Keye AR model and processor...")
    ar_model, ar_processor = train_rec.load_keye_ar(args.keye_ar_dir)
    ar_model.eval()

    # 3) Load Sana DiT model
    print("Loading Sana DiT model...")
    if args.dcp_source_dir and args.dcp_tag:
        # Convert DCP checkpoint to PyTorch
        train_rec.convert_dcp_checkpoint(
            source_dir=args.dcp_source_dir,
            target_dir=args.model_dir,
            tag=args.dcp_tag
        )

    # Load model config
    model_cfg = load_config(args.model_dir)
    model_class = get_model_class(model_cfg)
    model_for_vis = model_class.from_config(model_cfg)
    model_for_vis.load_checkpoint(args.model_dir)
    model_for_vis.eval()

    # Determine device and dtype
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # Move models to device
    vae = vae.to(device=device, dtype=dtype)
    ar_model = ar_model.to(device=device, dtype=dtype)
    model_for_vis = model_for_vis.to(device=device, dtype=dtype)

    # Set up image tokenizer
    image_tokenizer = ar_model

    # 4) Build dataset using provided dataset config (for processing helpers)
    with open(args.dataset_config, encoding='utf-8') as f:
        dataset_cfg = json.load(f)

    # Ensure processor_path is set to Keye AR if not present
    if not dataset_cfg.get('processor_path'):
        dataset_cfg['processor_path'] = args.keye_ar_dir

    dataset_cfg['image_size'] = args.image_size
    dataset_cfg['max_condition_length'] = args.max_condition_length
    # Pass rank/world_size so datasets expecting distributed info work in single-process mode

    dataset = GenEvalInferenceDataset(
        processor_path=args.keye_ar_dir, 
        gen_eval_csv_path=args.gen_eval_csv_path,
        infer_repeats=args.infer_repeats)

    # 5) Run DiT sampling pipeline *locally* and save results (DiT JPEGs + messages JSON)
    print("Running DiT sampling and saving results...")
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    import time

    latent_size = args.image_size // model_for_vis.config.vae_downsample_rate
    
    # Get latent channels from VAE config
    latent_channels = vae.config.latent_channels
    print(f"latent_channels: {latent_channels}")
    
    # Create data structures to store results
    images_dict = {}  # Key: sample index, Value: list of lists of PIL images
    samples_dict = {}  # Key: sample index, Value: original sample data

    for samples in dataset:
        samples = easydict.EasyDict(samples)
        # samples: 
        # {'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'a photo of a cow'}]}], 'metadata': {'index': 2, 'tag': 'single_object', 'include_class': 'cow', 'include_count': '1', 'include_color': None, 'include_position': None, 'exclude_class': None, 'exclude_count': None, 'question': 'a photo of a cow'}}

        with torch.no_grad():
            batch_size = samples.input_ids.shape[0]
            print(f"batch_size: {batch_size}")
            # Tokenize images to condition embeddings
            cond_embeds, cond_mask = tokenize_images(
                ar_model=image_tokenizer,
                batch_size=batch_size,
                max_condition_length=args.max_condition_length,
                input_ids=samples.input_ids.to(device=device),
                teacher_forcing=args.teacher_forcing,
                ar_processor=ar_processor,
            )

            # Prepare unconditional embeddings for CFG
            null_embed = model_for_vis.y_embedder.y_embedding
            seq_len = min(null_embed.shape[0], args.max_condition_length)
            uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            if seq_len < args.max_condition_length:
                padding = torch.zeros(
                    batch_size, args.max_condition_length - seq_len, uncond_embeds.shape[-1],
                    device=device, dtype=dtype
                )
                uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
            uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
            uncond_mask = torch.zeros(batch_size, args.max_condition_length, device=device)
            uncond_mask[:, :seq_len] = 1
            uncond_mask = uncond_mask[:, None, None, :]

            # Create scheduler and sample
            scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
            scheduler.set_timesteps(args.num_sampling_steps, device=device)

            generator = torch.Generator(device=device).manual_seed(args.seed)
            dit_latents = torch.randn(
                (batch_size, latent_channels, latent_size, latent_size),
                generator=generator,
                device=device,
                dtype=dtype,
            )

            cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
            mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)

            for t in scheduler.timesteps:
                print(f"dit_latents shape: {dit_latents.shape}")
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

            # Convert to PIL images and store in the dictionary
            sample_index = samples.metadata.index
            pil_images = []
            for i in range(batch_size):
                img_np = dit_np[i]
                img_pil = Image.fromarray((img_np * 255).astype('uint8'))
                pil_images.append([img_pil])  # Wrap in a list as per the required format
            
            # Add to images_dict
            if sample_index not in images_dict:
                images_dict[sample_index] = []
            images_dict[sample_index].extend(pil_images)
            
            # Add to samples_dict if not already present
            if sample_index not in samples_dict:
                # Convert easydict to regular dict and remove unnecessary fields
                sample_data = {
                    'messages': samples.messages,
                    'metadata': samples.metadata
                }
                samples_dict[sample_index] = sample_data

    # Save results in the required format after all inference is done
    print("Saving GenEval results...")
    ulmeval_dir = os.path.join(args.results_dir, 'ulmeval')
    os.makedirs(ulmeval_dir, exist_ok=True)
    
    # Save images as pickle file
    images_filename = f"{rank}{world_size}_GenEval.pkl"
    images_filepath = os.path.join(ulmeval_dir, images_filename)
    with open(images_filepath, 'wb') as f:
        pickle.dump(images_dict, f)
    
    # Save samples as json file
    samples_filename = f"{rank}{world_size}_GenEval.json"
    samples_filepath = os.path.join(ulmeval_dir, samples_filename)
    with open(samples_filepath, 'w', encoding='utf-8') as f:
        json.dump(samples_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {ulmeval_dir}")


if __name__ == "__main__":
    main()
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
import torch.distributed as dist
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing pretrained model or checkpoint")
    parser.add_argument("--dcp-ckpt-dir", type=str, default=None,
                        help="CKPT directory for DCP checkpoint conversion (required if --dcp-tag is used)")
    parser.add_argument("--dcp-tag", type=str, default=None,
                        help="Tag for DCP checkpoint (e.g., global_step8000)")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Optional model config JSON path (if not using --model-dir/config.json)")
    parser.add_argument("--model-config-overrides", type=str, nargs="*", default=[],
                        help="Override model config fields. Format: key=value. "
                             "Example: --model-config-overrides caption_channels=1024 model_max_length=324")
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
    parser.add_argument("--teacher-forcing", type=int, default=1,
                        help="Enable teacher forcing during inference")
    parser.add_argument("--port", type=int, default=29500,
                        help="Port to use for distributed communication")
    return parser.parse_args()


def setup_distributed_environment(rank: int = 0, world_size: int = 1, port:int = 29500) -> bool:
    """
    Initialize distributed environment for single-process inference runs using the
    same approach as our tests: TCP init on localhost (gloo backend).

    Returns:
        True if distributed was initialized successfully, otherwise False.
    """
    print(f"going to init: {rank}/{world_size}")
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://127.0.0.1:{port}',
        rank=rank,
        world_size=world_size,
    )
    print(f"Initialized local TCP-based distributed process group (127.0.0.1:{port})")
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
            **kwargs
        )
        print(f"after generate")
        embeddings = embeddings[0]
        return tokens, embeddings
        

def tokenize_images(ar_processor : AutoProcessor,
                    ar_model : KeyeARModel,
                    pixel_values: torch.Tensor,
                    image_grid_thw: torch.Tensor,
                    batch_size: int,
                    max_condition_length: int,
                    input_ids: Optional[torch.Tensor] = None,
                    cu_seqlens: Optional[torch.Tensor] = None,
                    teacher_forcing: bool = False,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize images using KeyeARModel.
    
    Args:
        ar_model: KeyeARModel instance
        pixel_values: Pixel values tensor [num_total_patches, ...]
        image_grid_thw: Grid info tensor [B, 3] where each row is (t, h, w)
        batch_size: Batch size (number of packed sequences)
        max_condition_length: Maximum condition sequence length for padding
        input_ids: Input token IDs [1, total_seq_len] (packed sequences)
        cu_seqlens: Cumulative sequence lengths for flash attention
    
    Returns:
        Tuple of (embeddings, attention_mask):
        - embeddings: [B, max_condition_length, embed_dim]
        - attention_mask: [B, 1, 1, max_condition_length] with 1s for valid tokens, 0s for padding
    """
    import IPython
    assert input_ids.size(0) == 1, "input_ids must has batch size of 1, got {}".format(input_ids.size(0))
    assistant_start_ids = ar_processor.tokenizer.encode("<|im_start|>assistant") # [151644, 77091]
    # input_ids: [batch_size, total_seq_len]
    if not teacher_forcing:
        # find assistant_start_ids in input_ids and delete the tokens after
        # Convert assistant_start_ids to tensor and ensure same device as input_ids
        assistant_start_tensor = torch.tensor(assistant_start_ids, device=input_ids.device, dtype=input_ids.dtype)
        
        # Get the sequence lengths
        seq_len = input_ids.size(1)
        assistant_len = len(assistant_start_ids)
        assistant_start_idx = -1
        
        # Search for the complete assistant_start_ids sequence
        if seq_len >= assistant_len:
            for i in range(seq_len - assistant_len + 1):
                # Check if the current window matches assistant_start_ids
                window = input_ids[0, i:i+assistant_len]
                if torch.all(window == assistant_start_tensor):
                    assistant_start_idx = i + assistant_len
                    break
        
        if assistant_start_idx != -1:
            # Keep only the tokens before assistant_start_ids
            input_ids = input_ids[:, :assistant_start_idx]
            print(f"Found assistant_start_ids at index {assistant_start_idx}, truncating input_ids to shape {input_ids.shape}")

    with torch.no_grad():
        
        # Create input_pos using cu_seqlens if provided
        if cu_seqlens is not None:
            # Calculate input_pos based on cu_seqlens
            # cu_seqlens: [0, seq_len1, seq_len1+seq_len2, ...]
            input_pos = []
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]
                pos_ids = torch.arange(seq_len, device=pixel_values.device, dtype=torch.long)
                input_pos.append(pos_ids)
            input_pos = torch.cat(input_pos, dim=0).unsqueeze(0)  # [1, total_seq_len]
        else:
            # Fallback: create input_pos from input_ids shape
            input_pos = torch.arange(input_ids.shape[1], device=pixel_values.device, dtype=torch.long).unsqueeze(0)

        # embeddings = outputs # .last_hidden_state  # [B, seq_len, embed_dim]
        input_ids, embeddings = get_model_embedding_and_tokens(
            model=ar_model,
            teacher_forcing=teacher_forcing,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_pos=input_pos,
            cu_seqlens=cu_seqlens,
            max_new_tokens=max_condition_length+4, # space,vis_start,vis_tok,vis_end,eos
        )

        # Extract embeddings between vision_start_id and vision_end_id
        vision_start_id = ar_model.config.qwen_config.vision_start_token_id
        vision_end_id = ar_model.config.qwen_config.vision_end_token_id
        
        # Find the positions of vision_start_id and vision_end_id in input_ids
        # input_ids shape is [1, total_seq_len] in packing case
        vision_start_mask = (input_ids == vision_start_id)
        vision_end_mask = (input_ids == vision_end_id)
        
        # For packing case: input_ids shape is [1, total_seq_len]
        # We need to find all vision_start_id and vision_end_id pairs in the sequence
        vision_embeddings_list = []
        vision_seq_lens = []
        
        # Get the flat input_ids (remove batch dimension for packing case)
        flat_input_ids = input_ids.squeeze(0)  # [total_seq_len]
        
        # Find all start and end positions
        start_positions = torch.nonzero(vision_start_mask.squeeze(0), as_tuple=True)[0]
        end_positions = torch.nonzero(vision_end_mask.squeeze(0), as_tuple=True)[0]
        
        # Check if we have matching number of start and end positions
        if len(start_positions) != len(end_positions):
            raise ValueError(f"Mismatched number of vision_start_id ({len(start_positions)}) and vision_end_id ({len(end_positions)}) tokens")
        
        # Extract embeddings for each vision segment
        for start_pos, end_pos in zip(start_positions, end_positions):
            # Check if start_pos comes before end_pos
            if start_pos >= end_pos:
                raise ValueError(f"vision_start_id ({start_pos.item()}) should come before vision_end_id ({end_pos.item()})")
            
            # Extract embeddings for this segment
            # embeddings shape is [1, total_seq_len, embed_dim] in packing case
            vision_embeddings = embeddings[0, start_pos:end_pos+1, :]  # [segment_len, embed_dim]
            vision_embeddings_list.append(vision_embeddings)
            vision_seq_lens.append(vision_embeddings.shape[0])
        
        # Check if we extracted the correct number of segments
        if len(vision_embeddings_list) != batch_size:
            raise ValueError(f"Extracted {len(vision_embeddings_list)} segments but batch_size is {batch_size}")
        
        # Stack the embeddings and handle variable sequence lengths
        max_vision_seq_len = max(vision_seq_lens)
        embed_dim = embeddings.shape[2]
        processed_embeddings = torch.zeros(batch_size, max_vision_seq_len, embed_dim,
                                            device=embeddings.device, dtype=embeddings.dtype)
        
        # Create attention mask: 1 for valid tokens, 0 for padding
        attention_mask = torch.zeros(batch_size, max_vision_seq_len,
                                   device=embeddings.device, dtype=torch.long)
        
        for i, emb in enumerate(vision_embeddings_list):
            seq_len = emb.shape[0]
            processed_embeddings[i, :seq_len, :] = emb
            attention_mask[i, :seq_len] = 1
        
        # Handle padding to max_condition_length
        current_seq_len = processed_embeddings.shape[1]
        if current_seq_len < max_condition_length:
            # Pad to max_condition_length
            padding_embeddings = torch.zeros(batch_size, max_condition_length - current_seq_len, embed_dim,
                                           device=processed_embeddings.device, dtype=processed_embeddings.dtype)
            processed_embeddings = torch.cat([processed_embeddings, padding_embeddings], dim=1)
            
            # Extend attention mask with zeros for padding
            padding_mask = torch.zeros(batch_size, max_condition_length - current_seq_len,
                                     device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
        elif current_seq_len > max_condition_length:
            # Truncate to max_condition_length
            processed_embeddings = processed_embeddings[:, :max_condition_length, :]
            attention_mask = attention_mask[:, :max_condition_length]

        # Reshape attention_mask to [B, 1, 1, max_condition_length]
        attention_mask = attention_mask[:, None, None, :]

    return processed_embeddings, attention_mask


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = train_rec.get_torch_dtype(args.dtype) if hasattr(train_rec, 'get_torch_dtype') else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally initialize a local single-process distributed group for dataset compatibility
    setup_distributed_environment(args.rank, args.world_size, args.port)

    # Convert DCP checkpoint if needed
    model_dir = args.model_dir
     
    if args.dcp_tag:
        converted_model_dir = os.path.join(args.dcp_ckpt_dir, args.dcp_tag, "converted")
        if not os.path.exists(converted_model_dir):
            print(f"Converting DCP checkpoint from {args.dcp_ckpt_dir} to {converted_model_dir}, dcp_tag={args.dcp_tag}")
            # Call DCP to torch conversion
            dcp_to_torch_convert(
                checkpoint_dir=args.dcp_ckpt_dir,
                tag=args.dcp_tag,
                source_dir=model_dir
            )
        else:
            print(f"DCP checkpoint already converted to torch format at: {converted_model_dir}")

        # Update model_dir to the converted directory
        model_dir = converted_model_dir
        print(f"Converted DCP checkpoint available at: {model_dir}")

    # 1) Load model config and instantiate model for visualization
    if args.model_config:
        model_config = load_config(args.model_config)
    else:
        # Expect config.json in model dir
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Model config not found at {cfg_path}. Provide --model-config if needed.")
        model_config = load_config(cfg_path)
    
    # Apply model config overrides from command line
    if args.model_config_overrides:
        overrides = parse_config_overrides(args.model_config_overrides)
        print(f"Applying model config overrides: {overrides}")
        for key, value in overrides.items():
            if hasattr(model_config, key):
                old_value = getattr(model_config, key)
                setattr(model_config, key, value)
                print(f"  {key}: {old_value} -> {value}")
            else:
                raise ValueError(f"Unknown model config field: {key}")

    model_class_name = model_config.model_class
    model_cls = get_model_class(model_class_name)

    print(f"Creating visualization model: {model_class_name}")
    with train_rec.set_default_dtype(args.dtype), torch.device("cpu"):
        model_for_vis = model_cls(model_config)

    # 2) Try to load checkpoint/state dict from model_dir
    print(f"Loading checkpoint from {model_dir}")
    sd = train_rec.load_hf_checkpoint(model_dir)
    model_for_vis.load_state_dict(sd, strict=False)
    model_for_vis.to(device).bfloat16()
    model_for_vis.eval()

    # 3) Load VAE and Keye AR ar_model/processor
    print("Loading VAE...")
    vae = train_rec.load_vae(args.vae_dir, device=device, dtype=dtype)

    print("Loading Keye AR ar_model/processor...")

    image_tokenizer = train_rec.load_keye_ar(args.keye_ar_dir, device=device, dtype=args.dtype, output_last_hidden_states_only=False)
    # Ensure ar_model/model is on the intended device (Triton kernels expect CUDA tensors)
    
    ar_processor = AutoProcessor.from_pretrained(
        args.keye_ar_dir,
        trust_remote_code=True
    )

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
            cond_embeds, cond_mask = tokenize_images(
                ar_model=image_tokenizer,
                pixel_values=loaded.pixel_values.to(device=device),
                image_grid_thw=loaded.image_grid_thw.to(device=device),
                batch_size=loaded.batch_size,
                max_condition_length=args.max_condition_length,
                input_ids=loaded.input_ids.to(device=device),
                teacher_forcing=args.teacher_forcing,
                ar_processor=ar_processor,
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
                noise_pred = model_for_vis.forward_with_dpmsolver(latent_input, timestep, cond_embeds_cfg, mask=mask_cfg, is_y_connected=True)
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
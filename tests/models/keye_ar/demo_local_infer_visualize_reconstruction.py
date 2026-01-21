"""
Inference demo: visualize DiT reconstructions using Keye AR processor (单机版本)

This is a local version of infer_visualize_reconstruction.py without distributed computing requirements.

Usage example:
    python tests/models/keye_ar/demo_local_infer_visualize_reconstruction.py
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoProcessor
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

# Import DCP to torch converter
from muse.tools.dcp2torch import convert as dcp_to_torch_convert

# Reuse helpers from the training recipe
from recipes.sana import train_sana_ar_dit as train_rec
from muse.config import load_config
from muse.models import get_model_class
from muse.models.keye_ar import KeyeARModel
from muse.utils.common import parse_config_overrides
from muse.training.common import set_default_dtype


def get_model_embedding_and_tokens(
        model: KeyeARModel,
        teacher_forcing: bool = False,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
    if teacher_forcing:
        kwargs["tokens"] = input_ids
        model.set_output_hidden_states([len(model.model.model.layers)])
        outputs = model(**kwargs)
        embeddings = outputs # .last_hidden_state  # [B, seq_len, embed_dim]
        return input_ids, embeddings[0]
    else:
        if "input_pos" in kwargs:
            del kwargs["input_pos"]
        if "pixel_values" in kwargs:
            del kwargs["pixel_values"]
            del kwargs["image_grid_thw"]
        if "cu_seqlens" in kwargs:
            del kwargs["cu_seqlens"]

        model.set_output_hidden_states([len(model.model.model.layers)])
        try:
            tokens, embeddings = model.generate(
                input_ids=input_ids,
                top_k=1,
                **kwargs
            )
        except Exception as e:
            raise Exception(f"Error in generate: {e}, input_ids: {input_ids}, kwargs: {kwargs}")
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
    """Tokenize images using KeyeARModel. (单机版本)
    
    Args:
        ar_processor: Keye AR processor
        ar_model: KeyeARModel instance
        pixel_values: Pixel values tensor [num_total_patches, ...]
        image_grid_thw: Grid info tensor [B, 3] where each row is (t, h, w)
        batch_size: Batch size (number of packed sequences)
        max_condition_length: Maximum condition sequence length for padding
        input_ids: Input token IDs [1, total_seq_len] (packed sequences)
        cu_seqlens: Cumulative sequence lengths for flash attention
        teacher_forcing: Whether to use teacher forcing
    
    Returns:
        Tuple of (embeddings, attention_mask):
        - embeddings: [B, max_condition_length, embed_dim]
        - attention_mask: [B, 1, 1, max_condition_length] with 1s for valid tokens, 0s for padding
    """
    assert input_ids.size(0) == 1, "input_ids must has batch size of 1, got {}".format(input_ids.size(0))
    assistant_start_ids = ar_processor.tokenizer.encode("<|im_start|>assistant\n") # [151644, 77091]
    
    if not teacher_forcing:
        # find assistant_start_ids in input_ids and delete the tokens after
        assistant_start_tensor = torch.tensor(assistant_start_ids, device=input_ids.device, dtype=input_ids.dtype)
        
        seq_len = input_ids.size(1)
        assistant_len = len(assistant_start_ids)
        assistant_start_idx = -1
        
        if seq_len >= assistant_len:
            for i in range(seq_len - assistant_len + 1):
                window = input_ids[0, i:i+assistant_len]
                if torch.all(window == assistant_start_tensor):
                    assistant_start_idx = i + assistant_len
                    break
        
        if assistant_start_idx != -1:
            input_ids = input_ids[:, :assistant_start_idx]
            print(f"Found assistant_start_ids at index {assistant_start_idx}, truncating input_ids to shape {input_ids.shape}")

    # Extract embeddings between vision_start_id and vision_end_id
    vision_start_id = ar_model.config.qwen_config.vision_start_token_id
    vision_end_id = ar_model.config.qwen_config.vision_end_token_id

    if input_ids[0][-1].item() != vision_start_id:
        input_ids = torch.cat([input_ids, torch.tensor([[vision_start_id]]).to(input_ids)], 1)

    with torch.no_grad():
        input_pos = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)

        input_ids, embeddings = get_model_embedding_and_tokens(
            model=ar_model,
            teacher_forcing=teacher_forcing,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            input_pos=input_pos,
            cu_seqlens=cu_seqlens,
            max_new_tokens=max_condition_length+4+99, # space,vis_start,vis_tok,vis_end,eos
        )
        
        # Find the positions of vision_start_id and vision_end_id in input_ids
        vision_start_mask = (input_ids == vision_start_id)
        vision_end_mask = (input_ids == vision_end_id)
        
        vision_embeddings_list = []
        vision_seq_lens = []
        
        start_positions = torch.nonzero(vision_start_mask.squeeze(0), as_tuple=True)[0]
        end_positions = torch.nonzero(vision_end_mask.squeeze(0), as_tuple=True)[0]
        
        if len(start_positions) != len(end_positions):
            print(f"Mismatched number of vision_start_id ({len(start_positions)}) and vision_end_id ({len(end_positions)}) tokens")
            vision_embeddings_list.append(embeddings[0, -max_condition_length:, :])
            vision_seq_lens.append(max_condition_length)
        else:
            for start_pos, end_pos in zip(start_positions, end_positions):
                if start_pos >= end_pos:
                    raise ValueError(f"vision_start_id ({start_pos.item()}) should come before vision_end_id ({end_pos.item()})")
                
                vision_embeddings = embeddings[0, start_pos:end_pos+1, :]  # [segment_len, embed_dim]
                vision_embeddings_list.append(vision_embeddings)
                vision_seq_lens.append(vision_embeddings.shape[0])
        
        vision_embeddings_list = [emb.to(embeddings.device) for emb in vision_embeddings_list]
        
        if len(vision_embeddings_list) != batch_size:
            print(f"Extracted {len(vision_embeddings_list)} segments but batch_size is {batch_size}")
            vision_embeddings_list.append(embeddings[0, -max_condition_length:, :])
            vision_seq_lens.append(max_condition_length)
            
        max_vision_seq_len = max(vision_seq_lens)
        embed_dim = embeddings.shape[2]
        processed_embeddings = torch.zeros(batch_size, max_vision_seq_len, embed_dim,
                                            device=embeddings.device, dtype=embeddings.dtype)
        
        attention_mask = torch.zeros(batch_size, max_vision_seq_len,
                                   device=embeddings.device, dtype=torch.long)
        
        for i, emb in enumerate(vision_embeddings_list):
            seq_len = emb.shape[0]
            processed_embeddings[i, :seq_len, :] = emb
            attention_mask[i, :seq_len] = 1
        
        current_seq_len = processed_embeddings.shape[1]
        if current_seq_len < max_condition_length:
            padding_embeddings = torch.zeros(batch_size, max_condition_length - current_seq_len, embed_dim,
                                           device=processed_embeddings.device, dtype=processed_embeddings.dtype)
            processed_embeddings = torch.cat([processed_embeddings, padding_embeddings], dim=1)
            
            padding_mask = torch.zeros(batch_size, max_condition_length - current_seq_len,
                                     device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, padding_mask], dim=1)
        elif current_seq_len > max_condition_length:
            processed_embeddings = processed_embeddings[:, :max_condition_length, :]
            attention_mask = attention_mask[:, :max_condition_length]

        attention_mask = attention_mask[:, None, None, :]

    return processed_embeddings, attention_mask


def load_keye_ar_local(tokenizer_dir: str, device: torch.device, dtype: torch.dtype, output_last_hidden_states_only=True):
    """Local version of load_keye_ar without distributed computing."""
    from muse.models.keye_ar import KeyeARModel
    with set_default_dtype(dtype), torch.device(device):
        tokenizer = KeyeARModel.from_pretrained(tokenizer_dir).eval()
        # Remove distributed print
        print(f"tokenizer={tokenizer}")
        tokenizer.config.qwen_config.output_last_hidden_states_only = output_last_hidden_states_only
        tokenizer.model.model.output_last_hidden_states_only = output_last_hidden_states_only
        tokenizer.requires_grad_(False)
    return tokenizer


def forward_ar_model(
        ar_model,
        input_ids,
        pixel_values,
        image_grid_thw,
    ):

    # forward one sample
    input_pos = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        outputs = ar_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cu_seqlens=torch.tensor([0, input_ids.shape[1]]).to(input_ids.device),
            input_pos=input_pos,
        )
    assert outputs.shape == (*input_ids.shape, ar_model.config.tokenizer_config.n_q_tokens + 1, ar_model.config.qwen_config.vocab_size + ar_model.config.tokenizer_config.codebook_size)
    print(f"outputs={outputs.shape}")


    # forward two samples in packing
    input_ids2 = torch.cat([input_ids, input_ids], dim=1) # torch.Size([1, 712])
    pixel_values2 = torch.cat([pixel_values, pixel_values], dim=0) # torch.Size([2592, 3, 14, 14])
    image_grid_thw2 = torch.cat([image_grid_thw, image_grid_thw], dim=0)   # b x 3
    cu_seqlens2 = torch.tensor([0, input_ids2.shape[1], input_ids2.shape[1] * 2]).to(input_ids2.device)
    input_pos2 = torch.cat([input_pos, input_pos], dim=-1)
    with torch.no_grad():
        outputs2 = ar_model(
            input_ids=input_ids2,
            pixel_values=pixel_values2,
            image_grid_thw=image_grid_thw2,
            cu_seqlens=cu_seqlens2,
            input_pos=input_pos2,
        )

    print(f"outputs2={outputs2.shape}")
    import IPython
    IPython.embed()


def main():
    # 直接在脚本中定义所有配置参数
    # 参考bash脚本 examples/sana/ar_dit/inference/run_local_infer_visualize_reconstruction.sh 中的默认值
    class Config:
        def __init__(self):
            self.model_dir = "/mmu_mllm_hdd_2/zangdunju/output2/RecoVLM/DiTSFT/batch6_324_1024_more_data/global_step80000/muse_converted/"
            self.dcp_ckpt_dir = None
            self.dcp_tag = None
            self.model_config = None
            self.model_config_overrides = []
            self.vae_dir = "/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/"
            self.keye_ar_dir = "/mmu_mllm_hdd_2/zhouyang12/output/Keye/vqar_11.7/run_8b_vis_stage3.29_1e-4/step18000/global_step18000/muse_converted"
            self.dataset_config = "examples/sana/ar_dit/exp21_ar_dit_324tokens_1e-4_reproduce_inf.json"
            self.parquet_path = "/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet"
            self.num_images = 1
            self.device = "cuda"
            self.dtype = "bfloat16"
            self.cfg_scale = 1.0
            self.num_sampling_steps = 50
            self.flow_shift = 3.0
            self.max_condition_length = 324
            self.image_size = 1024
            self.seed = 42
            self.results_dir = "./vis_output_local/results"
            self.teacher_forcing = 0
            self.linspace_sigmas = True
            self.savings = "/llm_reco/lingzhixin/recovlm_data/for_debug/infer_visualize_reconstruction/for_new_compare_v2.pt"
    
    args = Config()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = train_rec.get_torch_dtype(args.dtype) if hasattr(train_rec, 'get_torch_dtype') else torch.float32

    os.makedirs(args.results_dir, exist_ok=True)

    # Convert DCP checkpoint if needed
    model_dir = args.model_dir
    
    if args.dcp_tag:
        converted_model_dir = os.path.join(args.dcp_ckpt_dir, args.dcp_tag, "converted")  # pyright: ignore[reportCallIssue]
        if not os.path.exists(converted_model_dir):
            print(f"Converting DCP checkpoint from {args.dcp_ckpt_dir} to {converted_model_dir}, dcp_tag={args.dcp_tag}")
            dcp_to_torch_convert(
                checkpoint_dir=args.dcp_ckpt_dir,
                tag=args.dcp_tag,
                source_dir=model_dir
            )
        else:
            print(f"DCP checkpoint already converted to torch format at: {converted_model_dir}")

        model_dir = converted_model_dir
        print(f"Converted DCP checkpoint available at: {model_dir}")

    # Load model config and instantiate model for visualization
    if args.model_config:
        model_config = load_config(args.model_config)
    else:
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

    # Load checkpoint/state dict from model_dir
    print(f"Loading checkpoint from {model_dir}")
    sd = train_rec.load_hf_checkpoint(model_dir)
    model_for_vis.load_state_dict(sd, strict=False)
    model_for_vis.to(device).bfloat16()
    model_for_vis.eval()

    # Load VAE and Keye AR model/processor
    print("Loading VAE...")
    vae = train_rec.load_vae(args.vae_dir, device=device, dtype=dtype)

    print("Loading Keye AR model/processor...")
    # Use local version instead of distributed one
    image_tokenizer = load_keye_ar_local(args.keye_ar_dir, device=device, dtype=args.dtype, output_last_hidden_states_only=False)
    
    ar_processor = AutoProcessor.from_pretrained(
        args.keye_ar_dir,
        trust_remote_code=True
    )

    image_tokenizer = image_tokenizer.to(device).bfloat16()

    # Build dataset using provided dataset config
    with open(args.dataset_config, encoding='utf-8') as f:
        dataset_cfg = json.load(f)

    if not dataset_cfg.get('processor_path'):
        dataset_cfg['processor_path'] = args.keye_ar_dir

    dataset_cfg['image_size'] = args.image_size
    dataset_cfg['max_condition_length'] = args.max_condition_length
    
    # 设置单机模式参数
    dataset_cfg['rank'] = 0
    dataset_cfg['world_size'] = 1


    print(f"Building Chat2ImageDataset for visualization with config: {dataset_cfg}")
    dataset = train_rec.Chat2ImageDataset(**dataset_cfg)

    given_samples = [
        {
            "messages": [
                {"role": "user", "content": "Generate an image of a cat."},
            ],
            "images": {
            }
        }
    ]

    # Run DiT sampling pipeline locally and save results
    print("Running DiT sampling and saving results...")
    savings = {}  # pyright: ignore[reportUnusedVariable]
    with torch.no_grad():
        # Load samples / preprocess
        loaded = train_rec.VisReconstructionLoader()(
            args.parquet_path,
            dataset,
            args.image_size,
            device,
            dtype,
            args.num_images,
            tb_writer=None,
            vae=vae,
            given_samples=given_samples
        )

        # forward_ar_model(
        #     ar_model=image_tokenizer,
        #     input_ids=loaded.input_ids.to(device=device),
        #     pixel_values=loaded.pixel_values.to(device=device),
        #     image_grid_thw=loaded.image_grid_thw.to(device=device),
        # )

        # Tokenize images to condition embeddings
        cond_embeds, cond_mask = tokenize_images(
            ar_processor=ar_processor,
            ar_model=image_tokenizer,
            pixel_values=loaded.pixel_values.to(device=device),
            image_grid_thw=loaded.image_grid_thw.to(device=device),
            batch_size=loaded.batch_size,
            max_condition_length=args.max_condition_length,
            input_ids=loaded.input_ids.to(device=device),
            teacher_forcing=args.teacher_forcing,
        )
        savings["cond_embeds"] = cond_embeds
        savings["cond_mask"] = cond_mask

        print(f"loaded.pixel_values={loaded.pixel_values.shape}")
        print(f"cond_embeds={cond_embeds.shape}, cond_mask={cond_mask.shape}")
        cond_embeds = model_for_vis.diffusion_connector(cond_embeds)
        savings["connected_cond_embeds"] = cond_embeds
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

        if args.linspace_sigmas:
            sigmas = np.linspace(1.0, 1 / args.num_sampling_steps, args.num_sampling_steps)
            savings["sigmas"] = torch.from_numpy(sigmas).to(device=device, dtype=dtype)
            scheduler.set_timesteps(args.num_sampling_steps, sigmas=sigmas, device=device)
        else:
            scheduler.set_timesteps(args.num_sampling_steps, device=device)

        savings["seed"] = torch.tensor(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        dit_latents = torch.randn(
            (loaded.batch_size, loaded.latent_channels, loaded.latent_size, loaded.latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        savings["dit_latents"] = dit_latents
        
        print(f"uncond_embeds={uncond_embeds.shape}, cond_embeds={cond_embeds.shape}")

        cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
        mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)

        for t in scheduler.timesteps:
            latent_input = torch.cat([dit_latents] * 2)
            timestep = t.expand(latent_input.shape[0])
            noise_pred = model_for_vis.forward_with_dpmsolver(latent_input, timestep, cond_embeds_cfg, mask=mask_cfg, is_y_connected=True)
            savings[f"solved_noise_pred_{t}"] = noise_pred
            
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + args.cfg_scale * (noise_cond - noise_uncond)
            dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]
            savings[f"dit_latents_{t}"] = dit_latents
            savings[f"latent_input_{t}"] = dit_latents
            

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

        torch.save(savings, args.savings)
        print(f"Saved {args.savings}")

if __name__ == "__main__":
    main()
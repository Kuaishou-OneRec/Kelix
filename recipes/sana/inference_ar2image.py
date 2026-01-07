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
import numpy as np
import tqdm
import argparse
import os
import json
import torch
import easydict
import pickle
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
from muse.data.datasets.image import GenEvalInferenceDataset
from muse.training.parallel import (
    initialize_model_parallel
)
import pandas as pd
import glob
import csv
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory containing pretrained model or checkpoint (required for inference mode)")
    parser.add_argument("--mode", type=str, default='inference',
                        choices=["inference", "visualize"], help="Mode to run in")
    parser.add_argument("--dcp-ckpt-dir", type=str, default=None,
                        help="CKPT directory for DCP checkpoint conversion (required if --dcp-tag is used)")
    parser.add_argument("--dcp-tag", type=str, default=None,
                        help="Tag for DCP checkpoint (e.g., global_step8000)")
    parser.add_argument("--model-config", type=str, default=None,
                        help="Optional model config JSON path (if not using --model-dir/config.json)")
    parser.add_argument("--model-config-overrides", type=str, nargs="*", default=[],
                        help="Override model config fields. Format: key=value. "
                             "Example: --model-config-overrides caption_channels=1024 model_max_length=324")
    parser.add_argument("--vae-dir", type=str, default=None,
                        help="VAE directory (required for inference mode)")
    parser.add_argument("--keye-ar-dir", type=str, default=None,
                        help="Keye AR processor/model directory (required for inference mode)")
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
    parser.add_argument("--cond-pos-scale", type=float, default=1.0,
                        help="Scale factor for condition position embeddings")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Directory to save generated DiT JPEGs and messages JSON")

    parser.add_argument("--teacher-forcing", type=int, default=1,
                        help="Enable teacher forcing during inference")
    
    parser.add_argument("--tb-log-name", type=str, default="tg_log",
                        help="Log name for result aggregation and tensor board")
    
    parser.add_argument("--n_infer_items", type=int, default=999999,
                        help="Number of items to infer")
    parser.add_argument("--model-tag", type=str, default="BLIP3OTransformersSFT",
                        help="Tag for model checkpoint (e.g., global_step8000)")
    parser.add_argument("--eval-id", type=str, default="default",
                        help="Eval ID for GenEval results")
    parser.add_argument("--linspace-sigmas", action="store_true",
                        help="Use linspace sigmas for scheduler")
    parser.add_argument("--benchname", type=str, default="GenEval",
                        help="Benchmark name for result aggregation, GenEval|WISE_all|DPGBench")
    return parser.parse_args()


BENCHNAME2CSV_MAP = {
    "GenEval": "/llm_reco/lingzhixin/recovlm_data/generation_data/GenEval.tsv",
    "WISE_all": "/mmu_mllm_hdd_2/zangdunju/analysis/WISE/WISE_all.tsv",
    "DPGBench": "/mmu_mllm_hdd_2/zangdunju/analysis/DPGBench/DPG_Bench.tsv",
}
BENCHNAME2PROMPT_KEY_MAP = {
    "GenEval": "question",
    "WISE_all": "prompt",
    "DPGBench": "text",
}
BENCHNAME2INFER_REPEATS_MAP = {
    "GenEval": 4,
    "WISE_all": 1,
    "DPGBench": 4,
}


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
    """获取模型嵌入和token。
    
    Args:
        model: KeyeAR模型实例
        teacher_forcing: 是否使用teacher forcing模式
        input_ids: 输入token ID张量
        **kwargs: 其他模型参数
    
    Returns:
        tuple: (tokens, embeddings) - token序列和对应的嵌入表示
    """
    if teacher_forcing:
        # Teacher forcing模式：直接使用输入token
        kwargs["tokens"] = input_ids
        outputs = model(**kwargs)
        embeddings = outputs # .last_hidden_state  # [B, seq_len, embed_dim]
        model.set_output_hidden_states([len(model.model.model.layers)])
        return input_ids, embeddings
    else:
        # 自回归生成模式：移除不需要的参数
        if "input_pos" in kwargs:
            del kwargs["input_pos"]
        if "pixel_values" in kwargs:
            del kwargs["pixel_values"]
            del kwargs["image_grid_thw"]
        if "cu_seqlens" in kwargs:
            del kwargs["cu_seqlens"]

        model.set_output_hidden_states([len(model.model.model.layers)])
        try:
            # 生成token和嵌入
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
                    batch_size: int,
                    max_condition_length: int,
                    pixel_values: torch.Tensor = None,
                    image_grid_thw: torch.Tensor = None,
                    input_ids: Optional[torch.Tensor] = None,
                    cu_seqlens: Optional[torch.Tensor] = None,
                    teacher_forcing: bool = False,
                    keep_image_token_id_thresh: int = 999999999,
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
    
    >>> m = [{'role':'user','content':'hello'},{'role':'assistant', 'content': {'type': 'image', 'image':im}}]
    >>> proc.tokenizer.encode(proc.apply_chat_template(m))
    [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 14990, 151645, 198, 151644, 77091, 198, 151652, 151655, 151653, 151645, 198]
    >>> m = [{'role':'user','content':'hello'}]
    >>> proc.tokenizer.encode(proc.apply_chat_template(m))
    [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 14990, 151645, 198]
    """
    import IPython
    assert input_ids.size(0) == 1, "input_ids must has batch size of 1, got {}".format(input_ids.size(0))
    assistant_start_ids = ar_processor.tokenizer.encode("<|im_start|>assistant\n") # [151644, 77091]
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

    # Extract embeddings between vision_start_id and vision_end_id
    vision_start_id = ar_model.config.qwen_config.vision_start_token_id
    vision_end_id = ar_model.config.qwen_config.vision_end_token_id
    image_token_id = ar_model.config.qwen_config.image_token_id
    voc_size = ar_model.config.qwen_config.vocab_size

    if input_ids[0][-1].item() != vision_start_id:
        input_ids = torch.cat([input_ids, torch.tensor([[vision_start_id]]).to(input_ids)], 1)

    with torch.no_grad():
        input_pos = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)

        # embeddings = outputs # .last_hidden_state  # [B, seq_len, embed_dim]
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
        # input_ids shape is [1, total_seq_len] in packing case
        vision_start_mask = (input_ids == vision_start_id)
        vision_end_mask = (input_ids == vision_end_id)
        
        # For packing case: input_ids shape is [1, total_seq_len]
        # We need to find all vision_start_id and vision_end_id pairs in the sequence
        vision_embeddings_list = []
        vision_seq_lens = []
        
        # Get the flat input_ids (remove batch dimension for packing case)
        flat_input_ids = input_ids.squeeze(0)  # [total_seq_len]
        
        if flat_input_ids.ndim == 2:
            is_image_id = flat_input_ids[:,0] >= voc_size
        else:
            is_image_id = flat_input_ids == image_token_id


        # Find all start and end positions
        start_positions = torch.nonzero(vision_start_mask.squeeze(0), as_tuple=True)[0]
        end_positions = torch.nonzero(vision_end_mask.squeeze(0), as_tuple=True)[0]
        
        # Check if we have matching number of start and end positions
        if len(start_positions) != len(end_positions):
            print(f"Mismatched number of vision_start_id ({len(start_positions)}) and vision_end_id ({len(end_positions)}) tokens\ninput_ids:{input_ids}")
            vision_embeddings = embeddings[0, -max_condition_length:, :]
            vision_embeddings = vision_embeddings[is_image_id[-max_condition_length:], :]  # [valid_len, embed_dim]
            vision_embeddings_list.append(vision_embeddings)
            vision_seq_lens.append(max_condition_length)
        else:
            # Extract embeddings for each vision segment
            for start_pos, end_pos in zip(start_positions, end_positions):
                # Check if start_pos comes before end_pos
                if start_pos >= end_pos:
                    raise ValueError(f"vision_start_id ({start_pos.item()}) should come before vision_end_id ({end_pos.item()})")
                
                # Extract embeddings for this segment
                # embeddings shape is [1, total_seq_len, embed_dim] in packing case
                vision_embeddings = embeddings[0, start_pos:end_pos+1, :]  # [segment_len, embed_dim]
                vision_embeddings = vision_embeddings[is_image_id[start_pos:end_pos+1], :]  # [valid_len, embed_dim]
                vision_embeddings_list.append(vision_embeddings)
                vision_seq_lens.append(vision_embeddings.shape[0])
        
        vision_embeddings_list = [emb.to(embeddings.device) for emb in vision_embeddings_list]
        # Check if we extracted the correct number of segments
        if len(vision_embeddings_list) != batch_size:
            vision_embeddings = embeddings[0, -max_condition_length:, :]
            vision_ids = flat_input_ids[-max_condition_length:]
            vision_embeddings = vision_embeddings[is_image_id[-max_condition_length:], :]  # [valid_len, embed_dim]
            vision_embeddings_list.append(vision_embeddings)
            vision_seq_lens.append(max_condition_length)
            
        
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


def vae_encode(vae, images: torch.Tensor) -> torch.Tensor:
    """Encode images to latent space.
    
    Reference: Sana/diffusion/model/builder.py vae_encode for AutoencoderDC
    """
    with torch.no_grad():
        # VAE runs in float32 for precision, images should already be float32
        # Use indexing [0] which works for both tuple and EncoderOutput
        z = vae.encode(images)[0]
        z = z * vae.config.scaling_factor
    return z


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = train_rec.get_torch_dtype(args.dtype) if hasattr(train_rec, 'get_torch_dtype') else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    # Optionally initialize a local single-process distributed group for dataset compatibility
    setup_distributed_environment()

    # Convert DCP checkpoint if needed
    model_dir = args.model_dir
     
    if args.dcp_tag:
        converted_model_dir = os.path.join(args.dcp_ckpt_dir, args.dcp_tag, "converted")
        if not os.path.exists(converted_model_dir) and torch.distributed.get_rank() == 0:
            print(f"Converting DCP checkpoint from {args.dcp_ckpt_dir} to {converted_model_dir}, dcp_tag={args.dcp_tag}")
            # Call DCP to torch conversion
            dcp_to_torch_convert(
                checkpoint_dir=args.dcp_ckpt_dir,
                tag=args.dcp_tag,
                source_dir=model_dir
            )
            print("Conversion complete.")

        torch.distributed.barrier()

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

    latent_channels = vae.config.latent_channels

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

    benchmark_csv_path = BENCHNAME2CSV_MAP.get(args.benchname, None)
    if benchmark_csv_path is not None:
        dataset_cfg["gen_eval_csv_path"] = benchmark_csv_path
        dataset_cfg["prompt_key"] = BENCHNAME2PROMPT_KEY_MAP[args.benchname]
        dataset_cfg["infer_repeats"] = BENCHNAME2INFER_REPEATS_MAP[args.benchname]
        print(f"Using benchmark CSV: {benchmark_csv_path}")

    dataset = GenEvalInferenceDataset(
        # processor_path=args.keye_ar_dir, 
        **dataset_cfg
        )

    # 5) Run DiT sampling pipeline *locally* and save results (DiT JPEGs + messages JSON)
    print("Running DiT sampling and saving results...")
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    import time

    latent_size = args.image_size // model_for_vis.config.vae_downsample_rate
    # Create data structures to store results
    images_dict = {}  # Key: sample index, Value: list of lists of PIL images
    samples_dict = {}  # Key: sample index, Value: original sample data

    for i_sample, samples in tqdm.tqdm(enumerate(dataset)):
        if i_sample >= args.n_infer_items:
            break

        samples = easydict.EasyDict(samples)
        # samples: 
        # {'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': [{'type': 'text', 'text': 'a photo of a cow'}]}], 'metadata': {'index': 2, 'tag': 'single_object', 'include_class': 'cow', 'include_count': '1', 'include_color': None, 'include_position': None, 'exclude_class': None, 'exclude_count': None, 'question': 'a photo of a cow'}}

        with torch.no_grad():
            batch_size = samples.input_ids.shape[0]
            # Tokenize images to condition embeddings
            cond_embeds, cond_mask = tokenize_images(
                ar_model=image_tokenizer,
                batch_size=batch_size,
                max_condition_length=args.max_condition_length,
                input_ids=samples.input_ids.to(device=device),
                teacher_forcing=args.teacher_forcing,
                ar_processor=ar_processor,
                keep_image_token_id_thresh=image_tokenizer.config.qwen_config.vocab_size
            )
            cond_embeds = model_for_vis.diffusion_connector(cond_embeds)
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
            if args.linspace_sigmas:
                sigmas = np.linspace(1.0, 1 / args.num_sampling_steps, args.num_sampling_steps)
                scheduler.set_timesteps(args.num_sampling_steps, sigmas=sigmas, device=device)
            else:
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

            pos_args = train_rec.compute_pos_args(
                latent_hw=(latent_size, latent_size), 
                image_grid_thw=torch.tensor([1, args.max_condition_length**0.5, args.max_condition_length**0.5])[None], 
                max_seq_len=args.max_condition_length, 
                device=device, 
                cond_pos_scale=args.cond_pos_scale)
            
            model_kwargs={
                **pos_args,
                "is_y_connected": True,
            }
            
            for t in scheduler.timesteps:
                latent_input = torch.cat([dit_latents] * 2)
                timestep = t.expand(latent_input.shape[0])
                noise_pred = model_for_vis.forward_with_dpmsolver(
                    latent_input, 
                    timestep, 
                    cond_embeds_cfg, 
                    mask=mask_cfg,
                    **model_kwargs
                    )
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + args.cfg_scale * (noise_cond - noise_uncond)
                dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]

            # Decode DiT latents
            dit_recon_latents = dit_latents / vae.config.scaling_factor
            dit_recon_images = vae.decode(dit_recon_latents).sample
            dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)

            # Save DiT JPEGs and messages JSON
            # results_step_dir = os.path.join(args.output_dir, "step_0")
            # os.makedirs(results_step_dir, exist_ok=True)

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
    print(f"Saving {args.benchname} results...")
    ulmeval_dir = os.path.join(args.output_dir, 'ulmeval', "subresults")
    ulmeval_agg_dir = os.path.join(args.output_dir, 'ulmeval', "aggresults")
    os.makedirs(ulmeval_dir, exist_ok=True)
    os.makedirs(ulmeval_agg_dir, exist_ok=True)
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Save images as pickle file
    images_filename = f"{rank}{world_size}_{args.benchname}.pkl"
    images_filepath = os.path.join(ulmeval_dir, images_filename)
    with open(images_filepath, 'wb') as f:
        pickle.dump(images_dict, f)
    
    # Save samples as json file
    samples_filename = f"{rank}{world_size}_{args.benchname}.json"
    samples_filepath = os.path.join(ulmeval_dir, samples_filename)
    with open(samples_filepath, 'w', encoding='utf-8') as f:
        json.dump(samples_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {ulmeval_dir}")

    torch.distributed.barrier()
    
    # Aggregate results from all subresults files
    if torch.distributed.get_rank() == 0:

        
        # Create aggresults directory if not exists
        agg_output_dir = os.path.join(args.output_dir, 'ulmeval', "aggresults", args.model_tag, args.eval_id)
        os.makedirs(agg_output_dir, exist_ok=True)
        
        # Find all JSON and PKL files in subresults
        json_files = glob.glob(os.path.join(ulmeval_dir, f"*{args.benchname}.json"))
        pkl_files = glob.glob(os.path.join(ulmeval_dir, f"*{args.benchname}.pkl"))
        
        # Initialize DataFrame with the required columns
        columns = [
            "index", "tag", "include_class", "include_count", 
            "include_color", "include_position", "exclude_class", 
            "exclude_count", "question", "questions", "text",
            "item_id", "discipline", "explanation", "prompt"
            "prediction", 
        ]
        df = pd.DataFrame(columns=columns)
        
        # Process each JSON-PKL pair
        for json_file, pkl_file in zip(sorted(json_files), sorted(pkl_files)):
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                samples_data = json.load(f)
            
            # Load PKL data
            with open(pkl_file, 'rb') as f:
                images_data = pickle.load(f)
            
            # Combine data into DataFrame rows
            for sample_idx, sample in samples_data.items():
                metadata = sample.get('metadata', {})
                sample_key = int(sample_idx)
                
                # Get prediction images directly from images_data without conversion
                prediction_images = images_data.get(sample_key, None)
                
                row = {
                    "index": int(metadata.get('index', 0)),
                    "tag": metadata.get('tag', ''),
                    "include_class": metadata.get('include_class', ''),
                    "include_count": metadata.get('include_count', 0),
                    "include_color": metadata.get('include_color', ''),
                    "include_position": metadata.get('include_position', ''),
                    "exclude_class": metadata.get('exclude_class', ''),
                    "exclude_count": metadata.get('exclude_count', 0),
                    "question": metadata.get('question', ''),
                    "questions": metadata.get('questions', ''),
                    "text": metadata.get('text', ''),
                    "item_id": metadata.get('item_id', ''),
                    "discipline": metadata.get('discipline', ''),
                    "explanation": metadata.get('explanation', ''),
                    "prompt": metadata.get('prompt', ''),
                    "prediction": prediction_images  # Keep as original PIL Image list
                }
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        
        # Sort by index and save to pickle
        if 'index' in df: df = df.sort_values('index').reset_index(drop=True)
        output_pkl = os.path.join(agg_output_dir, f"{args.model_tag}_{args.benchname}.pkl")
        df.to_pickle(output_pkl)
        print(f"Aggregated results saved to: {output_pkl}")

def collect_eval_scores(dcp_ckpt_dir, model_tag="BLIP3OTransformersSFT", tb_log_name="tg_log"):
    """
    Collect 'all' evaluation scores from BLIP3OTransformersSFT_GenEval_score.csv files
    and write to TensorBoard and CSV.
    
    Args:
        dcp_ckpt_dir: Root directory containing checkpoint directories
        model_tag: Model tag for score file pattern (default: "BLIP3OTransformersSFT")
        tb_log_name: TensorBoard log name (default: "tg_log")
    
    Returns:
        None
    """
    if not dcp_ckpt_dir or not os.path.exists(dcp_ckpt_dir):
        print(f"Warning: DCP checkpoint directory {dcp_ckpt_dir} does not exist, skipping score collection")
        return
    
    # Create tf_eval_log directory for TensorBoard (clean up if exists)
    tf_eval_dir = os.path.join(dcp_ckpt_dir, "tf_eval_log")
    tb_writer = None
    
    try:
        # Remove existing TensorBoard logs to avoid duplicate entries
        if os.path.exists(tf_eval_dir):
            print(f"Removing existing TensorBoard logs: {tf_eval_dir}")
            import shutil
            shutil.rmtree(tf_eval_dir)
        
        os.makedirs(tf_eval_dir, exist_ok=True)
        
        # CSV file to store all scores (overwrite existing)
        csv_file = os.path.join(dcp_ckpt_dir, f"{args.benchname}_scores.csv")
        
        # TensorBoard writer
        tb_writer = SummaryWriter(log_dir=tf_eval_dir)
        
        # Find all score files
        score_pattern = os.path.join(dcp_ckpt_dir, "**", f"{model_tag}_{args.benchname}_score.csv")
        score_files = glob.glob(score_pattern, recursive=True)
        
        if not score_files:
            print(f"No score files found matching pattern: {score_pattern}")
            return
        
        scores_data = []
        
        for score_file in sorted(score_files):
            try:
                # Extract step number from file path
                # Path format: .../global_stepXXXXX/inference/GenEval/outputs/.../BLIP3OTransformersSFT_GenEval_score.csv
                path_parts = score_file.split('/')
                step_part = None
                for part in path_parts:
                    if part.startswith('global_step'):
                        step_part = part
                        break
                
                if not step_part:
                    print(f"Could not extract step from path: {score_file}")
                    continue
                    
                step = int(step_part.replace('global_step', ''))
                
                # Read the CSV file
                with open(score_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    
                    # Parse the all score - format: "0.65575"
                    all_score = None
                    for row in rows:
                        if len(row) >= 2 and row[0] == 'all':
                            try:
                                all_score = float(row[1].strip())
                                break
                            except ValueError:
                                continue
                    
                    if all_score is not None:
                        scores_data.append((step, all_score))
                        tb_writer.add_scalar(f'{args.benchname}/Overall_Score', all_score, step)
                        print(f"Step {step}: Overall Score = {all_score}")
                    else:
                        print(f"Could not parse overall score from {score_file}")
                        
            except Exception as e:
                print(f"Error processing {score_file}: {e}")
        
        # Sort by step and write to CSV
        if scores_data:
            scores_data.sort(key=lambda x: x[0])
            
            # Write to CSV with format: 
            # First row: Benchmark names
            # Subsequent rows: Step and corresponding scores for each benchmark
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # First row: benchmark names
                writer.writerow(['Step', f'{args.benchname}'])
                
                # Subsequent rows: step and scores
                for step, score in scores_data:
                    writer.writerow([step, score])
            
            print(f"Saved {len(scores_data)} scores to {csv_file}")
            print(f"TensorBoard logs saved to {tf_eval_dir}")
        else:
            print("No valid scores found")
    
    except Exception as e:
        print(f"Error in collect_eval_scores: {e}")
    finally:
        if tb_writer is not None:
            tb_writer.close()


# After inference completes, collect evaluation scores if dcp_ckpt_dir is provided
if __name__ == "__main__":
    args = parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'inference':
        # For inference mode, check required parameters
        if not args.model_dir:
            print("Error: --model-dir is required for inference mode")
            exit(1)
        if not args.vae_dir:
            print("Error: --vae-dir is required for inference mode")
            exit(1)
        if not args.keye_ar_dir:
            print("Error: --keye-ar-dir is required for inference mode")
            exit(1)
        main()
    else:
        # For visualize mode, only need dcp_ckpt_dir
        if not args.dcp_ckpt_dir:
            print("Error: --dcp-ckpt-dir is required for visualize mode")
            exit(1)
        collect_eval_scores(args.dcp_ckpt_dir, args.model_tag, args.tb_log_name)

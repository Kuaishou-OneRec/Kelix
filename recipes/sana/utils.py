import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional
from pathlib import Path

from muse.utils.common import print_rank_0
from muse.training.common import set_default_dtype

def load_vae(vae_dir: str, device: torch.device, dtype: torch.dtype):
    """Load VAE model from diffusers.
    
    Reference: Sana/diffusion/model/builder.py
    """
    from diffusers import AutoencoderDC
    
    print_rank_0(f"Loading VAE from {vae_dir}")
    vae = AutoencoderDC.from_pretrained(vae_dir, torch_dtype=dtype)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    
    return vae

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


def compute_input_pos(h: int, w: int, device: torch.device = None) -> dict:
    """Compute 2D position ids for RoPE from grid height and width.
    
    This function computes height and width position indices for each position
    in an h x w grid, suitable for 2D rotary positional embeddings.
    
    Args:
        h: Grid height (number of patches in height dimension)
        w: Grid width (number of patches in width dimension)
        device: Target device for tensors (optional)
    
    Returns:
        Dictionary with:
            - "height": Tensor of shape [h*w] with row indices (0 to h-1)
            - "width": Tensor of shape [h*w] with column indices (0 to w-1)
    
    Example:
        For a 2x3 grid, positions are laid out as:
            (0,0) (0,1) (0,2)
            (1,0) (1,1) (1,2)
        
        height_ids = [0, 0, 0, 1, 1, 1]
        width_ids  = [0, 1, 2, 0, 1, 2]
    """
    # Total number of positions
    seq_len = h * w
    
    # Create position indices
    pos_ids = torch.arange(seq_len, device=device)
    
    # Compute height (row) and width (column) indices
    height_ids = pos_ids // w  # Row index: 0, 0, 0, 1, 1, 1, ...
    width_ids = pos_ids % w    # Column index: 0, 1, 2, 0, 1, 2, ...
    
    return {"height": height_ids, "width": width_ids}


def load_image_tokenizer(tokenizer_dir: str,
            device: torch.device,
            dtype: torch.dtype,
            fusion_type: Literal["mean", "sum"] = "sum"):
    from muse.models.keye_tokenizer import KeyeImageTokenizer
    with set_default_dtype(dtype), torch.device(device):
        tokenizer = KeyeImageTokenizer.from_pretrained(
            tokenizer_dir, fusion_type=fusion_type).eval()
        tokenizer.requires_grad_(False)

    return tokenizer


def load_visualization_images(
    image_dir: str,
    processor,
    image_size: int,
    max_condition_length: int,
    device: torch.device,
    dtype: torch.dtype,
    num_images: Optional[int] = None
) -> tuple:
    """Load and preprocess images from a directory for visualization.
    
    Args:
        image_dir: Directory containing images (jpg, png, jpeg, webp)
        processor: AutoProcessor instance for image preprocessing
        image_size: Target image size
        max_condition_length: Maximum condition sequence length for image tokenizer
        device: Target device
        dtype: Target dtype
        num_images: Maximum number of images to load (None for all)
    
    Returns:
        Tuple of (original_images, pixel_values, image_grid_thw, vae_input_images)
        - original_images: List of PIL images (for visualization)
        - pixel_values: Tensor for image tokenizer
        - image_grid_thw: Grid info tensor for image tokenizer
        - vae_input_images: Tensor for VAE encoding [B, C, H, W] in [-1, 1]
    """
    from PIL import Image
    from torchvision import transforms
    from keye_vl_utils import process_vision_info
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []
    for f in sorted(os.listdir(image_dir)):
        if Path(f).suffix.lower() in image_extensions:
            image_files.append(os.path.join(image_dir, f))
    
    if num_images is not None:
        image_files = image_files[:num_images]
    
    if not image_files:
        print_rank_0(f"Warning: No images found in {image_dir}")
        return None, None, None, None
    
    print_rank_0(f"Loading {len(image_files)} images from {image_dir}")
    
    # Load images
    original_images = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        # Resize to target size
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        original_images.append(img)
    
    # Prepare messages format for processor (keye_vl_utils format)
    fake_messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
                "min_pixels": 4 * 28 * 28,
                "max_pixels": max_condition_length * 28 * 28
            } for img in original_images],
    }]
    text = processor.apply_chat_template(
        fake_messages,
        tokenize=False
    )
    # Process using keye_vl_utils
    image_inputs, _, _ = process_vision_info(fake_messages)
    
    # Use processor to get pixel_values and image_grid_thw
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    
    # Prepare images for VAE (normalize to [-1, 1])
    vae_transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    vae_input_images = torch.stack([vae_transform(img) for img in original_images])
    vae_input_images = vae_input_images.to(device=device, dtype=dtype)
    
    return original_images, pixel_values, image_grid_thw, vae_input_images


def load_fake_images_and_tokens(
    image_dir: str,
    processor,
    image_size: int,
    max_condition_length: int,
    device: torch.device,
    dtype: torch.dtype,
    num_images: Optional[int] = None,
    vocab_size: int = 151936,
    codebook_size: int = 65536,
    nq_tokens: int = 8,
    end_flags: list[int] = [151653, 151645],
) -> tuple:
    """Load and preprocess images from a directory for visualization.
    
    Args:
        image_dir: File containing generate images's prompts
        processor: AutoProcessor instance for image preprocessing
        image_size: Target image size
        max_condition_length: Maximum condition sequence length for image tokenizer
        device: Target device
        dtype: Target dtype
        num_images: Maximum number of images to load (None for all)
    
    Returns:
        Tuple of (original_images, pixel_values, image_grid_thw, vae_input_images)
        - original_images: List of PIL images (for visualization)
        - pixel_values: Tensor for image tokenizer
        - image_grid_thw: Grid info tensor for image tokenizer
        - vae_input_images: Tensor for VAE encoding [B, C, H, W] in [-1, 1]
    """
    from PIL import Image
    from torchvision import transforms
    from keye_vl_utils import process_vision_info
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

    index2tokens_dict = torch.load(image_dir, weights_only=False)
    
    tuple_tokens = [(key, index2tokens_dict[key]) for key in index2tokens_dict]
    tuple_tokens.sort(key=lambda x: x[0])
    
    keys = [x[0] for x in tuple_tokens]
    tokens = [x[1] for x in tuple_tokens]

    if num_images is not None:
        keys = keys[:num_images]
        tokens = tokens[:num_images]
    
    if not keys:
        print_rank_0(f"Warning: No tokens found in {image_dir}")
        return None, None, None, None
    
    tmp_tokens = list()
    tokens_length = list()

    for part in tokens:
        input_ids = part["input_ids"]
        generate_ids = part["generate_ids"]
        start = input_ids.shape[1]
        end = min(generate_ids.shape[1], start + max_condition_length)
        generate_ids = generate_ids[0, start: end, :-1]
        if end_flags is not None:
            end_length = len(end_flags)
            end_tensor = torch.LongTensor(end_flags)
            end_index = (generate_ids[:, 0].unfold(0, end_length, 1) == end_tensor).all(1).long().argmax()
            if end_index > 0:
                generate_ids = generate_ids[:end_index]
        indices_list = (generate_ids - vocab_size).unbind(-1)
        indices = torch.stack(
            [
                indices.clamp(i * (codebook_size // nq_tokens), (i + 1) * (codebook_size // nq_tokens) - 1) - i * (codebook_size // nq_tokens)
                for i, indices in enumerate(indices_list)
            ],
            dim=-1
        )
        tmp_tokens.append(indices)
        tokens_length.append(indices.shape[0])
    tokens = torch.concat(tmp_tokens, dim=0)
    
    print_rank_0(f"Loading {len(keys)} tokens from {image_dir}")

    # Load images
    original_images = list()
    for _ in keys:
        img = Image.new("RGB", (image_size, image_size), color="red")

        original_images.append(img)
    
    # Prepare messages format for processor (keye_vl_utils format)
    fake_messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
                "min_pixels": 4 * 28 * 28,
                "max_pixels": max_condition_length * 28 * 28
            } for img in original_images],
    }]
    text = processor.apply_chat_template(
        fake_messages,
        tokenize=False
    )
    # Process using keye_vl_utils
    image_inputs, _, _ = process_vision_info(fake_messages)
    
    # Use processor to get pixel_values and image_grid_thw
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt",
    )
    
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device=device)
    
    # Prepare images for VAE (normalize to [-1, 1])
    vae_transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])
    vae_input_images = torch.stack([vae_transform(img) for img in original_images])
    vae_input_images = vae_input_images.to(device=device, dtype=dtype)
    
    return original_images, pixel_values, image_grid_thw, vae_input_images, keys, tokens, tokens_length


@torch.no_grad()
def run_dit_reconstruction(
    model,
    vae,
    image_tokenizer,
    processor,
    image_dir: str,
    output_dir: str,
    cfg_scale: float,
    num_sampling_steps: int,
    flow_shift: float,
    max_condition_length: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int = 0,
    tb_writer=None,
    seed: int = 42,
    num_images: Optional[int] = None,
):
    """Run DiT reconstruction and save visualizations.
    
    Creates comparison images showing: Original | VAE Reconstruction | DiT Reconstruction
    Also saves individual images and optionally logs to TensorBoard.
    
    Args:
        model: DiT model
        vae: VAE model
        image_tokenizer: Image tokenizer
        processor: AutoProcessor for image preprocessing
        image_dir: Directory containing source images
        output_dir: Directory to save visualization results
        cfg_scale: CFG scale for sampling
        num_sampling_steps: Number of Euler sampling steps
        flow_shift: Flow shift parameter
        max_condition_length: Maximum condition sequence length
        image_size: Image size
        device: Device to run on
        dtype: Data type
        global_step: Current training step (used in output directory name, defaults to 0)
        tb_writer: TensorBoard SummaryWriter (optional)
        seed: Random seed for reproducibility
        num_images: Maximum number of images to process
    """
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    print_rank_0(f"[Step {global_step}] Running DiT reconstruction...")
    print_rank_0(f"  Input dir: {image_dir}")
    print_rank_0(f"  Output dir: {output_dir}")
    print_rank_0(f"  CFG scale: {cfg_scale}")
    print_rank_0(f"  Sampling steps: {num_sampling_steps}")
    print_rank_0(f"  Flow shift: {flow_shift}")
    
    # Load and preprocess images
    result = load_visualization_images(
        image_dir=image_dir,
        processor=processor,
        image_size=image_size,
        max_condition_length=max_condition_length,
        device=device,
        dtype=dtype,
        num_images=num_images,
    )
    
    if result[0] is None:
        print_rank_0("No images found, skipping...")
        return
    
    original_images, pixel_values, image_grid_thw, vae_input_images = result
    batch_size = len(original_images)
    print_rank_0(f"  Processing {batch_size} images...")
    
    # 1. VAE Reconstruction: encode -> decode
    print_rank_0("  VAE encoding...")
    latents = vae_encode(vae, vae_input_images)
    latent_channels = latents.shape[1]
    latent_size = latents.shape[2]
    
    print_rank_0("  VAE decoding (reconstruction)...")
    vae_recon_latents = latents / vae.config.scaling_factor
    vae_recon_images = vae.decode(vae_recon_latents).sample
    vae_recon_images = (vae_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 2. Get condition embeddings from image tokenizer
    print_rank_0("  Getting condition embeddings...")
    cond_embeds, cond_mask, max_seq_len = image_tokenizer.tokenize(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        max_pad_to=max_condition_length,
        return_attention_mask=True,
    )
    
    # Prepare unconditional embeddings using model's null embedding for CFG
    # Get the null embedding from model's y_embedder
    null_embed = model.y_embedder.y_embedding  # [token_num, caption_channels]
    # Truncate/pad to max_seq_len (dynamic padding) and expand to batch
    seq_len = min(null_embed.shape[0], max_seq_len)
    uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq_len, C]
    # Pad to max_seq_len if needed
    if seq_len < max_seq_len:
        padding = torch.zeros(
            batch_size, max_seq_len - seq_len, uncond_embeds.shape[-1],
            device=device, dtype=dtype
        )
        uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
    uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
    # Mask: mark the valid part of null embedding as 1
    uncond_mask = torch.zeros(batch_size, max_seq_len, device=device)
    uncond_mask[:, :seq_len] = 1
    uncond_mask = uncond_mask[:, None, None, :]  # [B, 1, 1, L]
    
    # 3. DiT sampling with Euler scheduler
    print_rank_0(f"  Euler sampling ({num_sampling_steps} steps, cfg={cfg_scale})...")
    
    # Create Euler scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_sampling_steps, device=device)
    
    # Initialize with random noise
    generator = torch.Generator(device=device).manual_seed(seed)
    dit_latents = torch.randn(
        (batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # Compute 2D position ids for RoPE
    # x_input_pos: for diffusion model's latent patches
    h_latent, w_latent = latent_size, latent_size
    x_input_pos = compute_input_pos(h_latent, w_latent, device=device)
    
    # cond_input_pos: for condition tokens from image tokenizer
    # image_grid_thw: [B, 3] where each row is (t, h, w)
    _, h_cond, w_cond = (image_grid_thw[0] // 2).tolist()
    cond_input_pos = compute_input_pos(int(h_cond), int(w_cond), device=device)

    # Pad cond_input_pos to max_seq_len (matching dynamic padding)
    cond_seq_len = int(h_cond * w_cond)
    pad_len = max_seq_len - cond_seq_len
    if pad_len > 0:
        cond_input_pos = {
            "height": F.pad(cond_input_pos["height"], (0, pad_len), value=0),
            "width": F.pad(cond_input_pos["width"], (0, pad_len), value=0),
        }
    
    # Prepare CFG inputs
    cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
    mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)
    
    # Euler sampling loop
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_input = torch.cat([dit_latents] * 2)
        timestep = t.expand(latent_input.shape[0])
        
        # Model prediction
        noise_pred = model.forward_with_dpmsolver(
            latent_input, timestep, cond_embeds_cfg, mask=mask_cfg,
            x_input_pos=x_input_pos,
            cond_input_pos=cond_input_pos,
        )
        
        # CFG combination
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        
        # Scheduler step
        dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]
    
    # Decode DiT latents to images
    print_rank_0("  Decoding DiT latents...")
    dit_recon_latents = dit_latents / vae.config.scaling_factor
    dit_recon_images = vae.decode(dit_recon_latents).sample
    dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 4. Create comparison images and save
    print_rank_0("  Saving comparison images...")
    vis_dir = os.path.join(output_dir, "visualization", f"step_{global_step}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    vae_recon_np = vae_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    dit_recon_np = dit_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    
    for i, orig_img in enumerate(original_images):
        # Get reconstructed images
        vae_img = Image.fromarray((vae_recon_np[i] * 255).round().astype("uint8"))
        dit_img = Image.fromarray((dit_recon_np[i] * 255).round().astype("uint8"))
        
        # Create side-by-side comparison: Original | VAE | DiT
        comparison = Image.new('RGB', (image_size * 3, image_size))
        comparison.paste(orig_img, (0, 0))
        comparison.paste(vae_img, (image_size, 0))
        comparison.paste(dit_img, (image_size * 2, 0))
        
        # Save comparison image
        comparison.save(os.path.join(vis_dir, f"comparison_{i}.png"))
        
        # Also save individual images
        orig_img.save(os.path.join(vis_dir, f"original_{i}.png"))
        vae_img.save(os.path.join(vis_dir, f"vae_recon_{i}.png"))
        dit_img.save(os.path.join(vis_dir, f"dit_recon_{i}.png"))
    
    # 5. Write to TensorBoard
    if tb_writer is not None:
        # Create a grid of all comparisons
        all_images = []
        for i, orig_img in enumerate(original_images):
            # Original
            orig_tensor = torch.from_numpy(np.array(orig_img)).permute(2, 0, 1).float() / 255.0
            all_images.append(orig_tensor)
            # VAE reconstruction
            all_images.append(vae_recon_images[i].cpu().float())
            # DiT reconstruction  
            all_images.append(dit_recon_images[i].cpu().float())
        
        # Stack and add to tensorboard
        grid = torch.stack(all_images)  # [N*3, C, H, W]
        from torchvision.utils import make_grid
        grid_img = make_grid(grid, nrow=3, padding=2)  # 3 images per row
        tb_writer.add_image(f"visualization/reconstruction", grid_img, global_step)
    
    print_rank_0(f"  Results saved to {vis_dir}")
    print_rank_0(f"  Processed {batch_size} images")



@torch.no_grad()
def run_vlm_reconstruction(
    model,
    vae,
    image_tokenizer,
    processor,
    image_dir: str,
    output_dir: str,
    cfg_scale: float,
    num_sampling_steps: int,
    flow_shift: float,
    max_condition_length: int,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int = 0,
    seed: int = 42,
    num_images: Optional[int] = None,
    num_generation_images: int = 1,
):
    """Run DiT reconstruction and save visualizations.
    
    Creates comparison images showing: Original | VAE Reconstruction | DiT Reconstruction
    Also saves individual images and optionally logs to TensorBoard.
    
    Args:
        model: DiT model
        vae: VAE model
        image_tokenizer: Image tokenizer
        processor: AutoProcessor for image preprocessing
        image_dir: Directory containing source images
        output_dir: Directory to save visualization results
        cfg_scale: CFG scale for sampling
        num_sampling_steps: Number of Euler sampling steps
        flow_shift: Flow shift parameter
        max_condition_length: Maximum condition sequence length
        image_size: Image size
        device: Device to run on
        dtype: Data type
        global_step: Current training step (used in output directory name, defaults to 0)
        tb_writer: TensorBoard SummaryWriter (optional)
        seed: Random seed for reproducibility
        num_images: Maximum number of images to process
    """
    from PIL import Image
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    print_rank_0(f"[Step {global_step}] Running DiT reconstruction...")
    print_rank_0(f"  Input dir: {image_dir}")
    print_rank_0(f"  Output dir: {output_dir}")
    print_rank_0(f"  CFG scale: {cfg_scale}")
    print_rank_0(f"  Sampling steps: {num_sampling_steps}")
    print_rank_0(f"  Flow shift: {flow_shift}")
    
    # Load and preprocess images
    result = load_fake_images_and_tokens(
        image_dir=image_dir,
        processor=processor,
        image_size=image_size,
        max_condition_length=max_condition_length,
        device=device,
        dtype=dtype,
        num_images=num_images,
    )
    
    if result[0] is None:
        print_rank_0("No images found, skipping...")
        return
    
    original_images, pixel_values, image_grid_thw, vae_input_images, keys, tokens, tokens_length = result
    batch_size = len(original_images)
    print_rank_0(f"  Processing {batch_size} images...")
    
    # 1. VAE Reconstruction: encode -> decode
    print_rank_0("  VAE encoding...")
    latents = vae_encode(vae, vae_input_images)
    latent_channels = latents.shape[1]
    latent_size = latents.shape[2]
    
    print_rank_0("  VAE decoding (reconstruction)...")
    vae_recon_latents = latents / vae.config.scaling_factor
    vae_recon_images = vae.decode(vae_recon_latents).sample
    vae_recon_images = (vae_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 2. Get condition embeddings from image tokenizer
    print_rank_0("  Getting condition embeddings...")
    cond_embeds, cond_mask, max_seq_len = image_tokenizer.tokenize_indices(
        indices=tokens,
        max_pad_to=max_condition_length,
        return_attention_mask=True,
        lengths=tokens_length,
    )

    dims = [num_generation_images] + [1 for _ in range(cond_embeds.ndim - 1)]
    cond_embeds = cond_embeds.repeat(*dims)
    dims = [num_generation_images] + [1 for _ in range(cond_mask.ndim - 1)]
    cond_mask = cond_mask.repeat(*dims)

    # Prepare unconditional embeddings using model's null embedding for CFG
    # Get the null embedding from model's y_embedder
    null_embed = model.y_embedder.y_embedding  # [token_num, caption_channels]
    # Truncate/pad to max_seq_len (dynamic padding) and expand to batch
    seq_len = min(null_embed.shape[0], max_seq_len)
    uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(batch_size * num_generation_images, -1, -1)  # [B, seq_len, C]
    # Pad to max_seq_len if needed
    if seq_len < max_seq_len:
        padding = torch.zeros(
            batch_size, max_seq_len - seq_len, uncond_embeds.shape[-1],
            device=device, dtype=dtype
        )
        uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
    uncond_embeds = uncond_embeds.to(device=device, dtype=dtype)
    # Mask: mark the valid part of null embedding as 1
    uncond_mask = torch.zeros(batch_size * num_generation_images, max_seq_len, device=device)
    uncond_mask[:, :seq_len] = 1
    uncond_mask = uncond_mask[:, None, None, :]  # [B, 1, 1, L]
    
    # 3. DiT sampling with Euler scheduler
    print_rank_0(f"  Euler sampling ({num_sampling_steps} steps, cfg={cfg_scale})...")
    
    # Create Euler scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    scheduler.set_timesteps(num_sampling_steps, device=device)
    
    # Initialize with random noise
    generator = torch.Generator(device=device).manual_seed(seed)
    dit_latents = torch.randn(
        (batch_size * num_generation_images, latent_channels, latent_size, latent_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # Compute 2D position ids for RoPE
    # x_input_pos: for diffusion model's latent patches
    h_latent, w_latent = latent_size, latent_size
    x_input_pos = compute_input_pos(h_latent, w_latent, device=device)
    
    # cond_input_pos: for condition tokens from image tokenizer
    # image_grid_thw: [B, 3] where each row is (t, h, w)
    _, h_cond, w_cond = (image_grid_thw[0] // 2).tolist()
    cond_input_pos = compute_input_pos(int(h_cond), int(w_cond), device=device)

    # Pad cond_input_pos to max_seq_len (matching dynamic padding)
    cond_seq_len = int(h_cond * w_cond)
    pad_len = max_seq_len - cond_seq_len
    if pad_len > 0:
        cond_input_pos = {
            "height": F.pad(cond_input_pos["height"], (0, pad_len), value=0),
            "width": F.pad(cond_input_pos["width"], (0, pad_len), value=0),
        }
    
    # Prepare CFG inputs
    cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
    mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)
    
    # Euler sampling loop
    for i, t in enumerate(scheduler.timesteps):
        # Expand latents for CFG
        latent_input = torch.cat([dit_latents] * 2)
        timestep = t.expand(latent_input.shape[0])
        
        # Model prediction
        noise_pred = model.forward_with_dpmsolver(
            latent_input, timestep, cond_embeds_cfg, mask=mask_cfg,
            x_input_pos=x_input_pos,
            cond_input_pos=cond_input_pos,
        )
        
        # CFG combination
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        
        # Scheduler step
        dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]
    
    # Decode DiT latents to images
    print_rank_0("  Decoding DiT latents...")
    dit_recon_latents = dit_latents / vae.config.scaling_factor
    dit_recon_images = vae.decode(dit_recon_latents).sample
    dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)
    
    # 4. Create comparison images and save
    print_rank_0("  Saving comparison images...")
    vis_dir = os.path.join(output_dir, "visualization", f"step_{global_step}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Convert tensors to numpy for visualization
    vae_recon_np = vae_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    dit_recon_np = dit_recon_images.cpu().permute(0, 2, 3, 1).float().numpy()
    
    for generation_index in range(num_generation_images):
        for i, (key, orig_img) in enumerate(zip(keys, original_images)):
            index = generation_index * batch_size + i
            # Get reconstructed images
            dit_img = Image.fromarray((dit_recon_np[index] * 255).round().astype("uint8"))
            
            # Save individual images
            dit_img.save(os.path.join(vis_dir, f"dit_recon_{key}_{generation_index}.png"))
    
    print_rank_0(f"  Results saved to {vis_dir}")
    print_rank_0(f"  Processed {batch_size} images")


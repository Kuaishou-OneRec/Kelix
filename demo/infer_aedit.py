"""Single-GPU Kelix-AEDIT inference: an image or local VQ indices -> PNG."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Support running the demo or a downloaded HF script inside a Kelix checkout.
for parent in (ROOT, *ROOT.parents):
    if (parent / 'muse' / '__init__.py').is_file():
        sys.path.insert(0, str(parent))
        break


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', default=str(ROOT) if (ROOT / 'dit').is_dir() else 'OpenOneRec/Kelix-AEDIT', help='Local ckpt directory or HF repo ID')
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image', help='Image to encode and reconstruct')
    inputs.add_argument('--tokens', help='Safely loaded .pt LongTensor [H*W,8], local codebook indices')
    p.add_argument('--grid', type=int, default=18, help='Condition grid width/height; 18 means 324 tokens')
    p.add_argument('--output', default='output.png')
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg', type=float, default=4.5)
    p.add_argument('--flow-shift', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    return p.parse_args()


def positions(h, w, device):
    import torch
    ids = torch.arange(h * w, device=device)
    return {'height': ids // w, 'width': ids % w}


def main(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffusers import AutoencoderDC, FlowMatchEulerDiscreteScheduler
    from transformers import AutoProcessor
    from muse.models.base import Model
    from muse.models.keye_tokenizer import KeyeImageTokenizer
    from muse.training.common import set_default_dtype

    if args.grid < 1 or args.steps < 1 or args.image_size < 32 or args.image_size % 32:
        raise ValueError('grid and steps must be positive; image-size must be a positive multiple of 32')
    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        from huggingface_hub import snapshot_download
        ckpt = Path(snapshot_download(args.ckpt))
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the FlashAttention vision encoder')
    torch.manual_seed(args.seed)
    with torch.inference_mode(), set_default_dtype(dtype):
        print('Loading DiT (strict=True)...', flush=True)
        model = Model.from_pretrained(str(ckpt / 'dit'), strict=True).to(device=device, dtype=dtype).eval()
        print('Loading image tokenizer (strict=True)...', flush=True)
        tokenizer = KeyeImageTokenizer.from_pretrained(
            str(ckpt / 'image_tokenizer'), strict=True, fusion_type='mean'
        ).to(device=device, dtype=dtype).eval()
        vae = AutoencoderDC.from_pretrained(str(ckpt / 'vae'), torch_dtype=dtype).to(device).eval()
        length = args.grid ** 2
        if args.tokens:
            indices = torch.load(args.tokens, map_location='cpu', weights_only=True)
            if not isinstance(indices, torch.Tensor) or tuple(indices.shape) != (length, tokenizer.n_q_tokens):
                raise ValueError(f'Expected integer tensor [{length}, {tokenizer.n_q_tokens}]')
            if indices.dtype not in (torch.int32, torch.int64):
                raise ValueError('Token indices must have integer dtype')
            limit = tokenizer.config.codebook_size // tokenizer.config.split_voc
            if indices.min() < 0 or indices.max() >= limit:
                raise ValueError(f'Expected per-codebook indices in [0, {limit}) (not AR vocabulary IDs)')
            cond, mask, seq_len = tokenizer.tokenize_indices(
                indices.to(device=device, dtype=torch.long), max_pad_to=length,
                lengths=[length], return_attention_mask=True,
            )
        else:
            processor = AutoProcessor.from_pretrained(
                str(ckpt / 'image_tokenizer'), trust_remote_code=True, use_fast=False,
            )
            # Match the original inference's square resize, then use an exact grid.
            # A merged Keye token covers 2*14 pixels; 18*28=504 for 324 tokens.
            image = Image.open(args.image).convert('RGB').resize(
                (args.image_size, args.image_size), Image.Resampling.LANCZOS,
            ).resize((args.grid * 28, args.grid * 28), Image.Resampling.BICUBIC)
            inputs = processor(text=[''], images=[image], return_tensors='pt')
            actual_grid = inputs['image_grid_thw'][0, 1:] // 2
            if actual_grid.tolist() != [args.grid, args.grid]:
                raise ValueError(f'Unexpected processor grid: {actual_grid.tolist()}')
            encoded = tokenizer(
                pixel_values=inputs['pixel_values'].to(device=device, dtype=dtype),
                image_grid_thw=inputs['image_grid_thw'].to(device),
            )
            features = encoded.get('token_embeds')
            if features is None:
                # Older Kelix releases compute fused features without returning them.
                features = torch.stack([
                    projector(z_q) for projector, z_q in
                    zip(tokenizer.up_projectors, encoded['z_q'])
                ], dim=1).mean(dim=1)
            if features.shape[0] != length:
                raise ValueError(f'Expected {length} image tokens, got {features.shape[0]}')
            cond, seq_len = features[None], length
            mask = torch.ones((1, 1, 1, length), device=device)
        if cond.shape[-1] != model.config.caption_channels:
            raise ValueError('Tokenizer condition dimension does not match the DiT config')
        null = model.y_embedder.y_embedding[:seq_len].to(device=device, dtype=dtype)
        valid = null.shape[0]
        null = F.pad(null, (0, 0, 0, seq_len - valid))[None]
        null_mask = torch.zeros_like(mask)
        null_mask[..., :valid] = 1
        cond_cfg = torch.cat([null, cond])
        mask_cfg = torch.cat([null_mask, mask])
        latent_size = args.image_size // 32
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latents = torch.randn((1, model.config.in_channels, latent_size, latent_size),
                              device=device, dtype=dtype, generator=generator)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=args.flow_shift)
        scheduler.set_timesteps(args.steps, device=device)
        x_pos = positions(latent_size, latent_size, device)
        y_pos = positions(args.grid, args.grid, device)
        print(f'Sampling {args.steps} steps...', flush=True)
        for t in scheduler.timesteps:
            pred = model.forward_with_dpmsolver(
                torch.cat([latents, latents]), t.expand(2), cond_cfg, mask=mask_cfg,
                x_input_pos=x_pos, cond_input_pos=y_pos,
            )
            uncond, conditional = pred.chunk(2)
            pred = uncond + args.cfg * (conditional - uncond)
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
        image = (vae.decode(latents / vae.config.scaling_factor).sample[0] / 2 + 0.5).clamp(0, 1)
        if not torch.isfinite(image).all():
            raise RuntimeError('Non-finite pixels encountered')
        pixels = (image.float().permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output)
        print(f'Saved {output.resolve()}')


if __name__ == '__main__':
    main(parse_args())

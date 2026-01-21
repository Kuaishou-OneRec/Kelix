"""Local inference demo: prompt -> image (PIL) for Sana AR-DiT + KeyeAR.

这个文件原本是一个“单机版本”的可运行脚本。
现在整理成一个类：
- 初始化负责加载：DiT 模型、VAE、KeyeAR tokenizer/processor、dataset（用于把 prompt 变成 input_ids 等）
- __call__(prompt) 负责：prompt -> input_ids -> DiT sampling -> VAE decode -> PIL.Image

新增：service 模式（可选）
- 当 LocalAR2ImageConfig.enable_service=True 时，调用 `gen.serve_forever()` 启动 HTTP 服务
- 客户端可 POST /generate，传入 JSON：
  - prompt: str（必填）
  - output_path: str（可选，目标图片写入地址）
- 服务端返回 JSON：
  - output_path: str（实际写入路径）

注意：
- 该类不依赖分布式。
- DCP 模式完全兼容原始逻辑：仅给 dcp_ckpt_dir + dcp_tag 时，仍需提供 source_model_dir 或 model_dir 作为 source_dir。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from transformers import AutoProcessor

from recipes.sana import train_sana_ar_dit as train_rec
from recipes.sana.inference_ar2image import tokenize_images

from muse.config import load_config
from muse.models import get_model_class
from muse.models.keye_ar import KeyeARModel
from muse.tools.dcp2torch import convert as dcp_to_torch_convert
from muse.training.common import set_default_dtype
from muse.utils.common import parse_config_overrides


def _get_torch_dtype(dtype: str) -> torch.dtype:
    if hasattr(train_rec, "get_torch_dtype"):
        return train_rec.get_torch_dtype(dtype)
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def _load_keye_ar_local(
    tokenizer_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    output_last_hidden_states_only: bool = False,
) -> KeyeARModel:
    """Local version of KeyeARModel.from_pretrained without distributed init."""
    with set_default_dtype(dtype), torch.device(device):
        tokenizer = KeyeARModel.from_pretrained(tokenizer_dir).eval()
        tokenizer.config.qwen_config.output_last_hidden_states_only = output_last_hidden_states_only
        tokenizer.model.model.output_last_hidden_states_only = output_last_hidden_states_only
        tokenizer.requires_grad_(False)
    return tokenizer


@dataclass
class LocalAR2ImageConfig:
# DiT model
    # - 常规模式：直接给 model_dir（通常是 converted/ 或 muse_converted/）
    # - DCP 模式：只给 dcp_ckpt_dir + dcp_tag 时，也必须给一个 source_model_dir（作为 dcp_to_torch_convert 的 source_dir）
    model_dir: Optional[str] = None
    source_model_dir: Optional[str] = None
    dcp_ckpt_dir: Optional[str] = None
    dcp_tag: Optional[str] = None

    model_config_overrides: Sequence[str] = ()

    # VAE & KeyeAR
    vae_dir: str = "/llm_reco_ssd/zhouyang12/models/SANA1.5_1.6B_1024px_diffusers/vae/"
    keye_ar_dir: str = ""

    # Dataset config（用来把 prompt -> input_ids）
    dataset_config: str = "examples/sana/ar_dit/exp21_ar_dit_324tokens_1e-4_reproduce_inf.json"
    parquet_path: str = ""  # 仅复用 loader 的预处理逻辑；given_samples 会覆盖真实内容

    # Sampling
    device: str = "cuda"
    dtype: str = "bfloat16"
    image_size: int = 1024
    seed: int = 42
    cfg_scale: float = 1.0
    num_sampling_steps: int = 50
    flow_shift: float = 3.0
    max_condition_length: int = 720
    linspace_sigmas: bool = True
    condition_on_special_tokens: bool = True

    # Service mode
    enable_service: bool = False
    service_host: str = "0.0.0.0"
    service_port: int = 18080
    service_output_dir: str = "./vis_output_local/service_outputs"


def load_local_ar2image_config(config_path: str) -> LocalAR2ImageConfig:
    """从 JSON 加载 LocalAR2ImageConfig。

    JSON 里只需要提供你想覆盖的字段即可；未提供的字段使用 dataclass 默认值。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data)}")
    return LocalAR2ImageConfig(**data)


class LocalAR2ImageGenerator:
    """单机 prompt -> image 的封装类（支持可选 service 模式）。"""

    def __init__(self, cfg: LocalAR2ImageConfig):
        self.cfg = cfg

        self.device = torch.device(cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.dtype = _get_torch_dtype(cfg.dtype)

        # service output dir
        self.service_output_dir = Path(cfg.service_output_dir)
        self.service_output_dir.mkdir(parents=True, exist_ok=True)

        # 1) resolve model_dir（兼容 DCP->torch 转换）
        model_dir = self._resolve_model_dir(cfg)

        # 2) load DiT config + model
        cfg_path = Path(model_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Model config not found at {cfg_path}")

        model_config = load_config(cfg_path)
        if cfg.model_config_overrides:
            overrides = parse_config_overrides(list(cfg.model_config_overrides))
            for k, v in overrides.items():
                if hasattr(model_config, k):
                    setattr(model_config, k, v)
                else:
                    raise ValueError(f"Unknown model config field: {k}")

        model_cls = get_model_class(model_config.model_class)

        with train_rec.set_default_dtype(cfg.dtype), torch.device("cpu"):
            self.dit = model_cls(model_config)

        sd = train_rec.load_hf_checkpoint(model_dir)
        self.dit.load_state_dict(sd, strict=False)
        self.dit.to(self.device).to(dtype=self.dtype)
        self.dit.eval()

        # 3) load VAE
        self.vae = train_rec.load_vae(cfg.vae_dir, device=self.device, dtype=self.dtype)

        # 4) load KeyeAR + processor
        self.keye_ar = _load_keye_ar_local(
            cfg.keye_ar_dir,
            device=self.device,
            dtype=self.dtype,
            output_last_hidden_states_only=False,
        )
        self.keye_ar = self.keye_ar.to(self.device).to(dtype=self.dtype)
        self.ar_processor = AutoProcessor.from_pretrained(cfg.keye_ar_dir, trust_remote_code=True)

        # 5) build dataset（单机 rank/world_size）
        with open(cfg.dataset_config, encoding="utf-8") as f:
            dataset_cfg = json.load(f)

        if not dataset_cfg.get("processor_path"):
            dataset_cfg["processor_path"] = cfg.keye_ar_dir
        dataset_cfg["image_size"] = cfg.image_size
        dataset_cfg["max_condition_length"] = cfg.max_condition_length
        dataset_cfg["rank"] = 0
        dataset_cfg["world_size"] = 1

        self.dataset = train_rec.Chat2ImageDataset(**dataset_cfg)

        # cached values for sampling
        self.latent_channels = self.vae.config.latent_channels
        self.latent_size = cfg.image_size // self.dit.config.vae_downsample_rate

    @staticmethod
    def _resolve_model_dir(cfg: LocalAR2ImageConfig) -> str:
        """兼容两种路径组织。

        保持原始脚本语义：只要 dcp_tag 非空就触发转换。
        """
        model_dir = cfg.model_dir

        if cfg.dcp_tag:
            if not cfg.dcp_ckpt_dir:
                raise ValueError("When dcp_tag is set, dcp_ckpt_dir must be provided")

            source_dir = cfg.source_model_dir or model_dir
            if not source_dir:
                raise ValueError(
                    "DCP mode requires a source model dir. Provide source_model_dir (preferred) or model_dir "
                    "as the source_dir for dcp_to_torch_convert."
                )

            converted_model_dir = os.path.join(cfg.dcp_ckpt_dir, cfg.dcp_tag, "converted")
            if not os.path.exists(converted_model_dir):
                print(f"Converting DCP checkpoint from {cfg.dcp_ckpt_dir} to {converted_model_dir}, dcp_tag={cfg.dcp_tag}")
                dcp_to_torch_convert(
                    checkpoint_dir=cfg.dcp_ckpt_dir,
                    tag=cfg.dcp_tag,
                    source_dir=source_dir,
                )
            else:
                print(f"DCP checkpoint already converted to torch format at: {converted_model_dir}")

            model_dir = converted_model_dir
            print(f"Converted DCP checkpoint available at: {model_dir}")

        if not model_dir:
            raise ValueError("model_dir is required (or use dcp_ckpt_dir+dcp_tag with source_model_dir)")

        return model_dir

    def _build_given_sample(self, prompt: str) -> Dict[str, Any]:
        # 对齐 bk.py：metadata / images 用一个占位即可（实际不会影响生成）。
        return {
            "uuid": "__xxxxxx__",
            "metadata": '{"images_info": {"output": {"width": 1024, "height": 781, "format": "PNG"}}}',
            "images": '{"output": "/mmu_mllm_hdd_2/lingzhixin/data/bytedance-research/UNO-1M/downloaded/images/split91/scene_prompt_object_object_v1_w1024_h2048_split_Stroller_Kiwi fruit_53519_asset0_scene5_1_781x1024.png"}',
            "videos": "{}",
            "source": "__default__",
            "messages": (
                '[{"role": "user", "content": [{"type": "text", "text": "Generate an image base on the description: __prompt__"}]},'
                '{"role": "assistant", "content": [{"type": "image", "image": "output"}]}]'
            ).replace("__prompt__", prompt),
        }

    @torch.no_grad()
    def __call__(self, prompt: str) -> Image.Image:
        """输入 prompt，返回一张 PIL.Image。"""
        given_samples = [self._build_given_sample(prompt)]

        loaded = train_rec.VisReconstructionLoader()(  # type: ignore[attr-defined]
            self.cfg.parquet_path,
            self.dataset,
            self.cfg.image_size,
            self.device,
            self.dtype,
            num_images=1,
            tb_writer=None,
            vae=self.vae,
            given_samples=given_samples,
            add_to_loaded=False,
        )

        cond_embeds, cond_mask, token_embed_lengths = tokenize_images(
            ar_processor=self.ar_processor,
            ar_model=self.keye_ar,
            batch_size=1,
            max_condition_length=self.cfg.max_condition_length,
            input_ids=loaded.input_ids.to(device=self.device),
            teacher_forcing=False,
            condition_on_special_tokens=self.cfg.condition_on_special_tokens,
        )

        cond_embeds = self.dit.diffusion_connector(cond_embeds)

        # unconditional embeds for CFG
        null_embed = self.dit.y_embedder.y_embedding
        seq_len = min(null_embed.shape[0], self.cfg.max_condition_length)
        uncond_embeds = null_embed[:seq_len, :].unsqueeze(0).expand(1, -1, -1)
        if seq_len < self.cfg.max_condition_length:
            padding = torch.zeros(
                1,
                self.cfg.max_condition_length - seq_len,
                uncond_embeds.shape[-1],
                device=self.device,
                dtype=self.dtype,
            )
            uncond_embeds = torch.cat([uncond_embeds, padding], dim=1)
        uncond_embeds = uncond_embeds.to(device=self.device, dtype=self.dtype)

        uncond_mask = torch.zeros(1, self.cfg.max_condition_length, device=self.device)
        uncond_mask[:, :seq_len] = 1
        uncond_mask = uncond_mask[:, None, None, :]

        scheduler = FlowMatchEulerDiscreteScheduler(shift=self.cfg.flow_shift)
        if self.cfg.linspace_sigmas:
            sigmas = np.linspace(1.0, 1 / self.cfg.num_sampling_steps, self.cfg.num_sampling_steps)
            scheduler.set_timesteps(self.cfg.num_sampling_steps, sigmas=sigmas, device=self.device)
        else:
            scheduler.set_timesteps(self.cfg.num_sampling_steps, device=self.device)

        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        dit_latents = torch.randn(
            (1, self.latent_channels, self.latent_size, self.latent_size),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
        mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)

        # pos args（和原流程保持一致）
        pos_args = train_rec.compute_pos_args(
            latent_hw=(self.latent_size, self.latent_size),
            image_grid_thw=torch.tensor(
                [1, 2 * self.cfg.max_condition_length**0.5, 2 * self.cfg.max_condition_length**0.5]
            )[None].to(self.device),
            max_seq_len=self.cfg.max_condition_length,
            device=self.device,
            cond_pos_scale=1.0,
            image_size=self.cfg.image_size,
            token_embed_lengths=token_embed_lengths,
        )
        model_kwargs = {**pos_args, "is_y_connected": True}

        for t in scheduler.timesteps:
            latent_input = torch.cat([dit_latents] * 2)
            timestep = t.expand(latent_input.shape[0])
            noise_pred = self.dit.forward_with_dpmsolver(
                latent_input,
                timestep,
                cond_embeds_cfg,
                mask=mask_cfg,
                **model_kwargs,
            )
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + self.cfg.cfg_scale * (noise_cond - noise_uncond)
            dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]

        dit_recon_latents = dit_latents / self.vae.config.scaling_factor
        dit_recon_images = self.vae.decode(dit_recon_latents).sample
        dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)

        img_np = dit_recon_images[0].detach().cpu().permute(1, 2, 0).float().numpy()
        return Image.fromarray((img_np * 255).round().astype("uint8"))

    def generate_to_path(self, prompt: str, output_path: Optional[str] = None) -> str:
        """生成图片并写入到 output_path（可缺省），返回实际写入路径。"""
        if output_path:
            out_path = Path(output_path)
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.service_output_dir / f"gen_{ts}_{uuid.uuid4().hex[:8]}.jpg"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = self(prompt)
        img.save(str(out_path), quality=95)
        return str(out_path)

    def serve_forever(self) -> None:
        """启动一个最轻量的 HTTP 服务（stdlib，无额外依赖）。"""

        generator = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/generate":
                    return self._send_json(404, {"error": "not found"})

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    req = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception as e:
                    return self._send_json(400, {"error": f"invalid json: {e}"})

                prompt = (req.get("prompt") or "").strip()
                output_path = req.get("output_path")

                print(f"receive request: prompt={prompt!r}, output_path={output_path!r}")

                if not prompt:
                    return self._send_json(400, {"error": "prompt is required"})

                try:
                    out_path = generator.generate_to_path(prompt=prompt, output_path=output_path)
                except Exception as e:
                    return self._send_json(500, {"error": str(e)})

                return self._send_json(200, {"output_path": out_path})

            def log_message(self, format: str, *args):  # noqa: A002
                return

        server = ThreadingHTTPServer((self.cfg.service_host, self.cfg.service_port), Handler)
        print(f"[LocalAR2ImageGenerator] serving on http://{self.cfg.service_host}:{self.cfg.service_port} (POST /generate)")
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to LocalAR2ImageConfig JSON")
    parser.add_argument("--prompt", type=str, default="a cat.")
    parser.add_argument("--output-path", type=str, default=None, help="Optional output image path when not serving")
    parser.add_argument("--cuda-visible-devices", type=str, default="1")
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)

    if args.config:
        cfg = load_local_ar2image_config(args.config)
    else:
        cfg = LocalAR2ImageConfig(
            model_dir="/mmu_mllm_hdd_2/lingzhixin/output/MuseV2/ar_dit/exp16x/exp168_0116sftv1_1e-4lr_sft_from162_49k/global_step4000/converted/",
            model_config_overrides=("model_max_length=720",),
            keye_ar_dir="/mmu_mllm_hdd_2/zhouyang12/output/Keye/sft_openmmreasoner/run_sft_exp11/step7000/global_step7000/muse_converted_fix/",
            dataset_config="examples/sana/ar_dit/exp21_ar_dit_324tokens_1e-4_reproduce_inf.json",
            parquet_path="/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet",
            max_condition_length=720,
            num_sampling_steps=50,
            linspace_sigmas=True,
            condition_on_special_tokens=True,
        )

    gen = LocalAR2ImageGenerator(cfg)

    if cfg.enable_service:
        gen.serve_forever()
        return

    out_path = gen.generate_to_path(prompt=args.prompt, output_path=args.output_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

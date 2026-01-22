"""Local inference demo: prompt -> image (PIL) for Sana AR-DiT + KeyeAR.

- 单卡模式：LocalAR2ImageGenerator（可 serve_forever）
- 多卡模式：MultiGPUAR2ImageService（round_robin / by_request）

支持两类可选请求参数：
1) LLM 生成参数：generation_args（透传到 tokenize_images(**generation_args)）
2) DiT 采样参数（覆盖 cfg 中同名字段）：
   - cfg_scale: float
   - num_sampling_steps: int
   - flow_shift: float
   - linspace_sigmas: bool

新增：返回 timing 信息（秒）：
- timings.total_time_s：服务端一次请求总耗时（从开始生成到图片落盘）
- timings.llm_time_s：tokenize_images（AR/LLM 推理部分）耗时
- timings.dit_time_s：DiT 采样循环耗时

Request JSON (POST /generate):
- prompt: str (required)
- output_path: str (optional)
- gpu_id: int (optional; multi-gpu)
- generation_args: dict (optional)
- cfg_scale: float (optional)
- num_sampling_steps: int (optional)
- flow_shift: float (optional)
- linspace_sigmas: bool (optional)

Response JSON:
- output_path: str
- gpu_id: int/str
- timings: { total_time_s: float, llm_time_s: float, dit_time_s: float }
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
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
    parquet_path: str = ""

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

    # Multi-GPU service
    service_gpu_ids: Optional[List[int]] = None
    service_gpu_policy: str = "round_robin"  # "round_robin" | "by_request"


def load_local_ar2image_config(config_path: str) -> LocalAR2ImageConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be an object, got: {type(data)}")
    return LocalAR2ImageConfig(**data)


class LocalAR2ImageGenerator:
    def __init__(self, cfg: LocalAR2ImageConfig):
        self.cfg = cfg

        self.device = torch.device(cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self.dtype = _get_torch_dtype(cfg.dtype)

        self.service_output_dir = Path(cfg.service_output_dir)
        self.service_output_dir.mkdir(parents=True, exist_ok=True)

        model_dir = self._resolve_model_dir(cfg)
        self.model_dir = model_dir

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

        self.vae = train_rec.load_vae(cfg.vae_dir, device=self.device, dtype=self.dtype)

        self.keye_ar = _load_keye_ar_local(
            cfg.keye_ar_dir,
            device=self.device,
            dtype=self.dtype,
            output_last_hidden_states_only=False,
        )
        self.keye_ar = self.keye_ar.to(self.device).to(dtype=self.dtype)
        self.ar_processor = AutoProcessor.from_pretrained(cfg.keye_ar_dir, trust_remote_code=True)

        with open(cfg.dataset_config, encoding="utf-8") as f:
            dataset_cfg = json.load(f)

        if not dataset_cfg.get("processor_path"):
            dataset_cfg["processor_path"] = cfg.keye_ar_dir
        dataset_cfg["image_size"] = cfg.image_size
        dataset_cfg["max_condition_length"] = cfg.max_condition_length
        dataset_cfg["rank"] = 0
        dataset_cfg["world_size"] = 1

        self.dataset = train_rec.Chat2ImageDataset(**dataset_cfg)

        self.latent_channels = self.vae.config.latent_channels
        self.latent_size = cfg.image_size // self.dit.config.vae_downsample_rate

    @staticmethod
    def _resolve_model_dir(cfg: LocalAR2ImageConfig) -> str:
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

            model_dir = converted_model_dir

        if not model_dir:
            raise ValueError("model_dir is required (or use dcp_ckpt_dir+dcp_tag with source_model_dir)")

        return model_dir

    def _build_given_sample(self, prompt: str) -> Dict[str, Any]:
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

    def serve_forever(self) -> None:
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
                generation_args = req.get("generation_args")

                # DiT generation args (optional overrides)
                cfg_scale = req.get("cfg_scale")
                num_sampling_steps = req.get("num_sampling_steps")
                flow_shift = req.get("flow_shift")
                linspace_sigmas = req.get("linspace_sigmas")

                if generation_args is not None and not isinstance(generation_args, dict):
                    return self._send_json(400, {"error": f"generation_args must be dict, got {type(generation_args)}"})

                for name, val in {
                    "cfg_scale": cfg_scale,
                    "num_sampling_steps": num_sampling_steps,
                    "flow_shift": flow_shift,
                }.items():
                    if val is not None and not isinstance(val, (int, float)):
                        return self._send_json(400, {"error": f"{name} must be number, got {type(val)}"})

                if num_sampling_steps is not None and not isinstance(num_sampling_steps, int):
                    return self._send_json(400, {"error": f"num_sampling_steps must be int, got {type(num_sampling_steps)}"})

                if linspace_sigmas is not None and not isinstance(linspace_sigmas, bool):
                    return self._send_json(400, {"error": f"linspace_sigmas must be bool, got {type(linspace_sigmas)}"})

                print(
                    f"[SingleGPU] request prompt={prompt!r} output_path={output_path!r} generation_args={generation_args!r} "
                    f"cfg_scale={cfg_scale!r} num_sampling_steps={num_sampling_steps!r} flow_shift={flow_shift!r} linspace_sigmas={linspace_sigmas!r}"
                )

                if not prompt:
                    return self._send_json(400, {"error": "prompt is required"})

                try:
                    ret = generator.generate_to_path(
                        prompt=prompt,
                        output_path=output_path,
                        generation_args=generation_args,
                        dit_sampling_overrides={
                            "cfg_scale": cfg_scale,
                            "num_sampling_steps": num_sampling_steps,
                            "flow_shift": flow_shift,
                            "linspace_sigmas": linspace_sigmas,
                        },
                    )
                except Exception as e:
                    return self._send_json(500, {"error": str(e)})

                return self._send_json(200, {**ret, "gpu_id": str(generator.device)})

            def log_message(self, format: str, *args):  # noqa: A002
                return

        server = ThreadingHTTPServer((self.cfg.service_host, self.cfg.service_port), Handler)
        print(f"[LocalAR2ImageGenerator] serving on http://{self.cfg.service_host}:{self.cfg.service_port} (POST /generate)")
        server.serve_forever()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        generation_args: Optional[Dict[str, Any]] = None,
        dit_sampling_overrides: Optional[Dict[str, Any]] = None,
        timings_out: Optional[Dict[str, float]] = None,
    ) -> Image.Image:
        generation_args = generation_args or {}
        dit_sampling_overrides = dit_sampling_overrides or {}

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

        if loaded is None or not hasattr(loaded, "input_ids") or loaded.input_ids is None:
            raise RuntimeError("VisReconstructionLoader failed to produce input_ids")

        t_llm0 = time.perf_counter()
        tokenized = tokenize_images(
            ar_processor=self.ar_processor,
            ar_model=self.keye_ar,
            batch_size=1,
            max_condition_length=self.cfg.max_condition_length,
            input_ids=loaded.input_ids.to(device=self.device),
            teacher_forcing=False,
            condition_on_special_tokens=self.cfg.condition_on_special_tokens,
            **generation_args,
        )
        # 类型提示：tokenize_images 在本工程中实际返回 (cond_embeds, cond_mask, token_embed_lengths)
        cond_embeds, cond_mask, token_embed_lengths = tokenized  # type: ignore[misc,assignment]
        t_llm1 = time.perf_counter()
        if timings_out is not None:
            timings_out["llm_time_s"] = t_llm1 - t_llm0

        if token_embed_lengths is None:
            # type: ignore[assignment]
            token_embed_lengths = torch.tensor([self.cfg.max_condition_length], device=self.device)

        cond_embeds = self.dit.diffusion_connector(cond_embeds)

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

        cfg_scale = float(dit_sampling_overrides.get("cfg_scale", self.cfg.cfg_scale))
        num_sampling_steps = int(dit_sampling_overrides.get("num_sampling_steps", self.cfg.num_sampling_steps))
        flow_shift = float(dit_sampling_overrides.get("flow_shift", self.cfg.flow_shift))
        linspace_sigmas = bool(dit_sampling_overrides.get("linspace_sigmas", self.cfg.linspace_sigmas))

        scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
        if linspace_sigmas:
            sigmas = np.linspace(1.0, 1 / num_sampling_steps, num_sampling_steps)
            scheduler.set_timesteps(num_sampling_steps, sigmas=sigmas, device=self.device)
        else:
            scheduler.set_timesteps(num_sampling_steps, device=self.device)

        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        dit_latents = torch.randn(
            (1, self.latent_channels, self.latent_size, self.latent_size),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        cond_embeds_cfg = torch.cat([uncond_embeds, cond_embeds], dim=0)
        mask_cfg = torch.cat([uncond_mask, cond_mask], dim=0)

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

        t_dit0 = time.perf_counter()
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
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            dit_latents = scheduler.step(noise_pred, t, dit_latents, return_dict=False)[0]
        t_dit1 = time.perf_counter()
        if timings_out is not None:
            timings_out["dit_time_s"] = t_dit1 - t_dit0

        dit_recon_latents = dit_latents / self.vae.config.scaling_factor
        dit_recon_images = self.vae.decode(dit_recon_latents).sample
        dit_recon_images = (dit_recon_images / 2 + 0.5).clamp(0, 1)

        img_np = dit_recon_images[0].detach().cpu().permute(1, 2, 0).float().numpy()
        return Image.fromarray((img_np * 255).round().astype("uint8"))

    def generate_to_path(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        dit_sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if output_path:
            out_path = Path(output_path)
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.service_output_dir / f"gen_{ts}_{uuid.uuid4().hex[:8]}.jpg"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        timings: Dict[str, float] = {}
        t_total0 = time.perf_counter()
        img = self(
            prompt,
            generation_args=generation_args,
            dit_sampling_overrides=dit_sampling_overrides,
            timings_out=timings,
        )
        img.save(str(out_path), quality=95)
        t_total1 = time.perf_counter()
        timings["total_time_s"] = t_total1 - t_total0

        return {"output_path": str(out_path.resolve()), "timings": timings}


class MultiGPUAR2ImageService:
    def __init__(self, base_cfg: LocalAR2ImageConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot start multi-gpu service")

        self.base_cfg = base_cfg

        if base_cfg.service_gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.gpu_ids = list(base_cfg.service_gpu_ids)

        if not self.gpu_ids:
            raise ValueError("service_gpu_ids is empty")

        self.policy = (base_cfg.service_gpu_policy or "round_robin").strip()
        if self.policy not in {"round_robin", "by_request"}:
            raise ValueError(f"Unknown service_gpu_policy: {self.policy}")

        self._rr_idx = 0
        self._rr_lock = threading.Lock()

        self.generators: Dict[int, LocalAR2ImageGenerator] = {}
        self.locks: Dict[int, threading.Lock] = {}

        print(f"[MultiGPUAR2ImageService] initializing on GPUs: {self.gpu_ids}, policy={self.policy}")

        for gid in self.gpu_ids:
            cfg = LocalAR2ImageConfig(**{**base_cfg.__dict__})
            cfg.device = f"cuda:{gid}"
            self.generators[gid] = LocalAR2ImageGenerator(cfg)
            self.locks[gid] = threading.Lock()
            print(f"[MultiGPUAR2ImageService] loaded generator on cuda:{gid}")

    def _pick_gpu(self, requested_gpu_id: Optional[int]) -> int:
        if requested_gpu_id is not None:
            if requested_gpu_id not in self.generators:
                raise ValueError(f"gpu_id={requested_gpu_id} not in service_gpu_ids={self.gpu_ids}")
            return requested_gpu_id

        if self.policy == "by_request":
            raise ValueError("gpu_id is required when service_gpu_policy=by_request")

        with self._rr_lock:
            gid = self.gpu_ids[self._rr_idx % len(self.gpu_ids)]
            self._rr_idx += 1
        return gid

    def generate(
        self,
        prompt: str,
        output_path: Optional[str],
        gpu_id: Optional[int],
        generation_args: Optional[Dict[str, Any]] = None,
        dit_sampling_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        gid = self._pick_gpu(gpu_id)
        generation_args = generation_args or {}

        print(
            f"[MultiGPUAR2ImageService] dispatch gpu={gid} prompt={prompt!r} output_path={output_path!r} generation_args={generation_args}"
        )

        gen = self.generators[gid]
        lock = self.locks[gid]

        with lock:
            ret = gen.generate_to_path(
                prompt=prompt,
                output_path=output_path,
                generation_args=generation_args,
                dit_sampling_overrides=dit_sampling_overrides,
            )

        return {**ret, "gpu_id": gid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to LocalAR2ImageConfig JSON")
    parser.add_argument("--prompt", type=str, default="a cat.")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--cuda-visible-devices", type=str, default="6,7")
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
            enable_service=True,
            service_gpu_ids=[0],
            service_gpu_policy="round_robin",
        )

    if cfg.enable_service and cfg.service_gpu_ids and len(cfg.service_gpu_ids) > 1:
        service = MultiGPUAR2ImageService(cfg)

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
                gpu_id = req.get("gpu_id")
                generation_args = req.get("generation_args")

                # DiT generation args (optional overrides)
                cfg_scale = req.get("cfg_scale")
                num_sampling_steps = req.get("num_sampling_steps")
                flow_shift = req.get("flow_shift")
                linspace_sigmas = req.get("linspace_sigmas")

                if generation_args is not None and not isinstance(generation_args, dict):
                    return self._send_json(400, {"error": f"generation_args must be dict, got {type(generation_args)}"})

                for name, val in {
                    "cfg_scale": cfg_scale,
                    "num_sampling_steps": num_sampling_steps,
                    "flow_shift": flow_shift,
                }.items():
                    if val is not None and not isinstance(val, (int, float)):
                        return self._send_json(400, {"error": f"{name} must be number, got {type(val)}"})

                if num_sampling_steps is not None and not isinstance(num_sampling_steps, int):
                    return self._send_json(400, {"error": f"num_sampling_steps must be int, got {type(num_sampling_steps)}"})

                if linspace_sigmas is not None and not isinstance(linspace_sigmas, bool):
                    return self._send_json(400, {"error": f"linspace_sigmas must be bool, got {type(linspace_sigmas)}"})

                if gpu_id is not None:
                    try:
                        gpu_id = int(gpu_id)
                    except Exception:
                        return self._send_json(400, {"error": f"gpu_id must be int, got {gpu_id!r}"})

                print(
                    f"receive request: prompt={prompt!r}, output_path={output_path!r}, gpu_id={gpu_id!r}, generation_args={generation_args!r}, "
                    f"cfg_scale={cfg_scale!r}, num_sampling_steps={num_sampling_steps!r}, flow_shift={flow_shift!r}, linspace_sigmas={linspace_sigmas!r}"
                )

                if not prompt:
                    return self._send_json(400, {"error": "prompt is required"})

                try:
                    ret = service.generate(
                        prompt=prompt,
                        output_path=output_path,
                        gpu_id=gpu_id,
                        generation_args=generation_args,
                        dit_sampling_overrides={
                            "cfg_scale": cfg_scale,
                            "num_sampling_steps": num_sampling_steps,
                            "flow_shift": flow_shift,
                            "linspace_sigmas": linspace_sigmas,
                        },
                    )
                except Exception as e:
                    return self._send_json(500, {"error": str(e)})

                return self._send_json(200, ret)

            def log_message(self, format: str, *args):  # noqa: A002
                return

        server = ThreadingHTTPServer((cfg.service_host, cfg.service_port), Handler)
        print(f"[MultiGPUAR2ImageService] serving on http://{cfg.service_host}:{cfg.service_port} (POST /generate)")
        server.serve_forever()
        return

    gen = LocalAR2ImageGenerator(cfg)

    if cfg.enable_service:
        gen.serve_forever()
        return

    ret = gen.generate_to_path(prompt=args.prompt, output_path=args.output_path)
    print(f"saved: {ret['output_path']}")
    print(f"timings: {ret['timings']}")


if __name__ == "__main__":
    main()

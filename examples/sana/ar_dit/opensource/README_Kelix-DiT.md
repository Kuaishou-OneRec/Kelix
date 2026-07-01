---
language:
  - en
library_name: "muse"
tags:
  - text-to-image
  - image-generation
  - diffusion
  - diffusion-transformer
  - DiT
  - flow-matching
  - unified-multimodal
  - discrete-token
  - kelix
base_model: "Efficient-Large-Model/Sana_1600M_1024px_diffusers"
pipeline_tag: text-to-image
---

# Kelix-DiT

[Paper](https://arxiv.org/pdf/2602.09843) | [Citation](#citation)

<p align="center">
  <img src="assert/fig3.png" alt="Kelix training pipeline: Kelix-Tok, Unified LLM, and Image DiT" width="90%">
</p>

<p align="center"><b>Figure 1:</b> The auto-regressive training workflow of Kelix, including the Kelix Tokenizer, the Unified LLM, and the Image DiT de-tokenizer.</p>

## Introduction

**Kelix-DiT** is the **pretraining-stage** checkpoint of the diffusion-based image de-tokenizer of **Kelix**, a fully discrete autoregressive unified multimodal model. It renders high-fidelity **1024×1024** images from the semantic hidden states produced by the Kelix unified LLM, closing the long-standing understanding gap between discrete and continuous visual representations.

Kelix is built on a modular *Tokenizer → LLM → Detokenizer* pipeline:

- **Kelix-Tok** — a multi-token vision tokenizer that decomposes each patch embedding into `N` parallel discrete codes, expanding the coding capacity exponentially while keeping the LLM context length unchanged via sum pooling on the encoder side.
- **Kelix-LLM** — a unified Qwen3-8B backbone trained with a **Next-Block Prediction (NBP)** paradigm.
- **Kelix-DiT** (this checkpoint) — a diffusion-based image de-tokenizer that turns the LLM's hidden states into high-resolution images.

Kelix achieves state-of-the-art results among comparable-scale unified models on both understanding and generation benchmarks; notably, it reaches **86.7 on OCRBench**, matching continuous-feature VLMs and surpassing the previous best discrete model by **+23%**.

## Model Overview

| Item | Value |
|---|---|
| Role | Diffusion-based image de-tokenizer (pretraining stage) |
| Base architecture | SANA-DiT (customized) |
| Training objective | Flow-matching |
| Latent VAE | DC-AE-F32C32 (32× spatial downsampling → 32×32 latent) |
| Condition | Last hidden states from the Kelix LLM (between vision start/end tokens) |
| Output resolution | **1024 × 1024** |
| Training stage | **Pretraining** (Stage 1 of 2) |

## Training (Pretraining Stage)

This checkpoint corresponds to the **Pretraining Stage** of Kelix-DiT:

- **Data**: inverted large-scale image-caption pairs.
- **Condition image**: resized to **504×504** for Kelix-LLM hidden-state extraction.
- **Target image**: resized to **1024×1024** and encoded into a 32×32 latent via the frozen DC-AE.
- **Aspect ratio**: source images constrained to 0.67–1.5 to preserve native composition.
- **Optimization**: all DiT parameters are updated; the Kelix LLM, DC-AE encoder, and DC-AE decoder are frozen.

This stage equips the de-tokenizer with robust semantic–image alignment and strong generalization to diverse, unseen scenarios. The companion **SFT-stage** checkpoint is released as [`OpenOneRec/Kelix-SFT`](https://huggingface.co/OpenOneRec/Kelix-SFT), which further enhances instruction-following and fine-grained control.

## Usage

Kelix-DiT is designed to be driven by the hidden states of the Kelix unified LLM. A typical generation pipeline is:

1. Feed a text prompt (and optional images) into the **Kelix LLM**, which autoregressively produces last hidden states `{h_*}` for the image blocks.
2. Use `{h_*}` as the semantic condition (y-embedder input) for Kelix-DiT.
3. Run flow-matching denoising in the DC-AE latent space (32×32) and decode with DC-AE to obtain a 1024×1024 image.

> ⚠️ This checkpoint is a **component** of the Kelix pipeline, not a standalone text-to-image model. To generate images end-to-end you also need the Kelix unified LLM and the frozen DC-AE-F32C32 VAE.

## Key Results

Kelix (8B, with Kelix-DiT) image-generation results:

| Benchmark | Score |
|---|---|
| GenEval (Overall) | **87.6** |
| WISE (Overall) | **57.0** |
| DPG-Bench (Overall) | **85.5** |

Highlights (see the technical report for full tables):
- GenEval **87.6** — SOTA among discrete-tokenization unified models, **+0.6** over the 27B Qwen-Image.
- WISE **57.0** — 2nd only to Nextflow (7B, 59.0), beating all continuous-tokenization unified models and larger dedicated T2I models (e.g., FLUX.1-dev 12B, 50.0).
- DPG-Bench **85.5** — competitive with the SOTA X-Omni (7B, 87.7) without using reinforcement learning.

## Citation

If you find Kelix useful, please cite our technical report.

```bibtex
@article{kelix2026,
  title   = {Kelix Technique Report: Closing the Understanding Gap of Discrete Tokens in Unified Multimodal Models},
  author  = {Kuaishou Technology},
  journal = {arXiv preprint arXiv:2602.09843},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.09843}
}
```

## License

Please contact the OneRec Team for the license of the Kelix series. The base SANA-DiT and DC-AE components are subject to their respective original licenses.

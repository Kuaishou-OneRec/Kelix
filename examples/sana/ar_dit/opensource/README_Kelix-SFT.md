---
language:
  - en
library_name: "muse"
tags:
  - unified-multimodal
  - discrete-token
  - image-understanding
  - text-to-image
  - image-generation
  - diffusion
  - diffusion-transformer
  - DiT
  - flow-matching
  - vision-language-model
  - VLM
  - next-block-prediction
  - kelix
  - sft
base_model: "Qwen/Qwen3-8B"
pipeline_tag: any-to-any
---

# Kelix-SFT

[Paper](https://arxiv.org/pdf/2602.09843) | [Citation](#citation)

<p align="center">
  <img src="assert/fig3.png" alt="Kelix training pipeline: Kelix-Tok, Unified LLM, and Image DiT" width="90%">
</p>

<p align="center"><b>Figure 1:</b> The auto-regressive training workflow of Kelix, including the Kelix Tokenizer, the Unified LLM, and the Image DiT de-tokenizer.</p>

## Introduction

**Kelix-SFT** is the **complete end-to-end release** of **Kelix** at the supervised-fine-tuning (SFT) stage — a fully discrete autoregressive unified multimodal model. This release bundles **all three components** of the Kelix pipeline:

1. **Kelix-Tok** — the multi-token discrete vision tokenizer (NaViT encoder + multi-token VQ codebooks),
2. **Kelix-LLM** — the unified Qwen3-8B backbone trained with Next-Block Prediction (NBP), with the vocabulary expanded by 65,536 visual entries,
3. **Kelix-DiT (SFT)** — the SFT-stage diffusion-based image de-tokenizer, fine-tuned from [`OpenOneRec/Kelix-DiT`](https://huggingface.co/OpenOneRec/Kelix-DiT).

Together they form a modular *Tokenizer → LLM → Detokenizer* system that unifies **multimodal understanding** and **image generation** under a single autoregressive objective. Use this repo if you want the full Kelix pipeline for both understanding and generation; use [`OpenOneRec/Kelix-DiT`](https://huggingface.co/OpenOneRec/Kelix-DiT) if you only need the pretraining-stage image de-tokenizer component.

Kelix achieves state-of-the-art results among comparable-scale unified models on both understanding and generation benchmarks; notably, it reaches **86.7 on OCRBench**, matching continuous-feature VLMs and surpassing the previous best discrete model by **+23%**.

## What's in this release

| Component | Description | Init / Base |
|---|---|---|
| **Kelix-Tok** | NaViT-style vision encoder + multi-token VQ (`N=8` independent sub-codebooks, total size `S=65,536`, sum-pooled on the encoder side). Encodes images into discrete visual codes aligned with the LLM text space. | NaViT initialized from Keye-VL 1.5; codebooks K-means initialized, trained with SimVQ |
| **Kelix-LLM** | Unified autoregressive backbone with Block Encoder / Block Decoder and a Next-Block Prediction (NBP) objective (text block size 2, visual block size `N+1=9`). | Qwen3-8B, vocabulary expanded by 65,536 visual entries |
| **Kelix-DiT (SFT)** | SFT-stage diffusion-based image de-tokenizer (flow-matching, SANA-DiT based), conditioned on the LLM's last hidden states. | Fine-tuned from `OpenOneRec/Kelix-DiT` |
| *(frozen, not included)* | DC-AE-F32C32 latent VAE (32× spatial downsampling → 32×32 latent) | Used as-is from the original DC-AE release |

> ⚠️ The **DC-AE-F32C32** VAE used to map pixels ↔ latents for Kelix-DiT is **frozen and not retrained**, so it is not duplicated in this release. Please load it from its original repo when running the image-generation path.

## Training (SFT Stage)

This release corresponds to the **Stage 4: Supervised Fine-Tuning** of the Kelix-LLM pipeline and the **SFT Stage** of the Kelix-DiT pipeline, building directly on the annealed Stage-3 checkpoints:

**Kelix-LLM (Stage 4 — SFT):**
- **Data**: replayed high-quality Stage-3 data, task-specific chain-of-thought (CoT) data, targeted image-generation data (attribute / subject / scene / style / perspective control), and task-specific enhancement data (complex OCR, math reasoning, STEM).
- **Filtering**: rejection sampling is retained to filter noisy samples and mitigate hallucinations.
- **Objective**: preserves the unified multimodal capability while boosting task-specific performance, keeping understanding and generation in balance.

**Kelix-DiT (SFT Stage):**
- **Data**: high-quality text-to-image data filtered via **rejection sampling** to remove low-quality pairs and mitigate hallucinations.
- **Fine-grained control**: curated to cover key control dimensions — attribute control (color, size), subject control (quantity, type), scene generation (indoor/outdoor, time of day), artistic style (oil painting, sketch), and perspective (top-down, close-up).
- **Target image**: strictly constrained to **1024×1024**.
- **Optimization**: all DiT parameters are updated; the Kelix LLM, DC-AE encoder, and DC-AE decoder are frozen.

Together, these significantly improve the model's adherence to diverse and precise textual instructions, enabling high-fidelity generation that strictly complies with complex prompts while maintaining strong multimodal understanding.

## Usage

This release is the full Kelix pipeline and can be used for **both understanding and generation** end-to-end (the only external dependency is the frozen DC-AE-F32C32 VAE for the generation path).

**Image understanding (VQA / OCR / reasoning):**
1. Encode the input image(s) with **Kelix-Tok** into `N`-way discrete visual codes, sum-pooled into a single composite token per patch on the encoder side.
2. Feed the visual codes + text tokens into **Kelix-LLM** (Block Encoder → NBP backbone → Block Decoder).
3. Decode the predicted text tokens via the standard text tokenizer.

**Image generation (text-to-image, 1024×1024):**
1. Feed a text prompt (and optional images) into **Kelix-LLM**, which autoregressively produces last hidden states `{h_*}` for the image blocks.
2. Use `{h_*}` as the semantic condition (y-embedder input) for **Kelix-DiT (SFT)**.
3. Run flow-matching denoising in the DC-AE latent space (32×32) and decode with the frozen DC-AE to obtain a 1024×1024 image.

> 💡 For broad-scenario generalization on the generation side, the pretraining-stage de-tokenizer [`OpenOneRec/Kelix-DiT`](https://huggingface.co/OpenOneRec/Kelix-DiT) can be swapped in for Kelix-DiT (SFT); Kelix-Tok and Kelix-LLM in this release are compatible with both.

## Key Results

Kelix (8B) results with this release:

**Image understanding**

| Benchmark | SEED-Bench | RealWorldQA | MMBench-EN | AI2D | MMMU | MathVista | ChartQA | TextVQA | OCRBench |
|---|---|---|---|---|---|---|---|---|---|
| Score | 76.0 | 72.1 | 80.2 | 82.4 | 54.1 | 76.5 | 83.0 | 81.4 | **86.7** |

**Image generation**

| Benchmark | GenEval (Overall) | WISE (Overall) | DPG-Bench (Overall) |
|---|---|---|---|
| Score | **87.6** | **57.0** | **85.5** |

Highlights (see the technical report for full tables):
- **OCRBench 86.7** — matching Qwen2.5-VL-7B (86.4), **+23%** over the previous best discrete unified model X-Omni (70.4). First discrete-token unified model to close the gap with continuous-feature VLMs on text-rich tasks.
- **MathVista 76.5** — surpasses the 14B Bagel (73.1) and the 30B Manzano (73.3) despite a much smaller parameter count.
- **GenEval 87.6** — SOTA among discrete-tokenization unified models, **+0.6** over the 27B Qwen-Image.
- **WISE 57.0** — 2nd only to Nextflow (7B, 59.0), beating all continuous-tokenization unified models and larger dedicated T2I models (e.g., FLUX.1-dev 12B, 50.0).
- **DPG-Bench 85.5** — competitive with the SOTA X-Omni (7B, 87.7) **without** using reinforcement learning.

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

Please contact the OneRec Team for the license of the Kelix series. The base components (Qwen3-8B, SANA-DiT, DC-AE, Keye-VL / NaViT) are subject to their respective original licenses.

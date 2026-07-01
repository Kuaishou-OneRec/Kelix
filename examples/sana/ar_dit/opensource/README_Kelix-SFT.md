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
  - instruction-following
  - kelix
  - sft
base_model: "OpenOneRec/Kelix-DiT"
pipeline_tag: text-to-image
arxiv: "https://arxiv.org/pdf/2602.09843"
---

# Kelix-SFT

**Kelix-SFT** is the **supervised-fine-tuning (SFT) stage** checkpoint of the diffusion-based image de-tokenizer of **Kelix**, a fully discrete autoregressive unified multimodal model proposed by the OneRec Team. Built on top of [`OpenOneRec/Kelix-DiT`](https://huggingface.co/OpenOneRec/Kelix-DiT), it is fine-tuned on high-quality, rejection-sampled text-to-image data to improve **instruction-following** and **fine-grained control** while still rendering **1024×1024** images.

> 📄 **Technical report**: <https://arxiv.org/pdf/2602.09843>

## Background

Most vision–language models (VLMs) rely on a hybrid interface — discrete text tokens paired with continuous ViT features — and are biased toward understanding. Fully autoregressive unified models that use **discrete** visual tokens, on the other hand, have historically suffered from an information bottleneck: a single discrete code carries far less information than the continuous embedding it replaces, which degrades multimodal understanding (especially on text-rich tasks such as OCRBench).

Kelix addresses this with a **multi-token vision tokenizer** (Kelix-Tok) that decomposes each patch embedding into `N` parallel discrete codes, expanding the coding capacity exponentially while keeping the LLM context length unchanged via sum pooling on the encoder side. The unified LLM (Qwen3-8B) is trained with a **Next-Block Prediction (NBP)** paradigm, and a diffusion-based **image de-tokenizer (Kelix-DiT)** turns the LLM's hidden states into high-resolution images — forming a modular *Tokenizer → LLM → Detokenizer* pipeline that unifies understanding and generation under a single autoregressive objective.

Kelix achieves state-of-the-art results among comparable-scale unified models on both understanding and generation benchmarks; notably, it reaches **86.7 on OCRBench**, matching continuous-feature VLMs and surpassing the previous best discrete model by **+23%**.

## Model Overview

| Item | Value |
|---|---|
| Role | Diffusion-based image de-tokenizer (SFT stage) |
| Base architecture | SANA-DiT (customized), fine-tuned from `OpenOneRec/Kelix-DiT` |
| Training objective | Flow-matching |
| Latent VAE | DC-AE-F32C32 (32× spatial downsampling → 32×32 latent) |
| Condition | Last hidden states from the Kelix LLM (between vision start/end tokens) |
| Output resolution | **1024 × 1024** (strictly constrained) |
| Training stage | **SFT** (Stage 2 of 2) |

## Training (SFT Stage)

This checkpoint corresponds to the **SFT Stage** of Kelix-DiT, building directly on the pretraining-stage checkpoint:

- **Data**: high-quality text-to-image data filtered via **rejection sampling** to remove low-quality pairs and mitigate hallucinations.
- **Fine-grained control**: curated to cover key control dimensions — attribute control (color, size), subject control (quantity, type), scene generation (indoor/outdoor, time of day), artistic style (oil painting, sketch), and perspective (top-down, close-up).
- **Target image**: strictly constrained to **1024×1024**.
- **Optimization**: all DiT parameters are updated; the Kelix LLM, DC-AE encoder, and DC-AE decoder are frozen.

This stage significantly improves the model's adherence to diverse and precise textual instructions, enabling high-fidelity generation that strictly complies with complex prompts.

## Usage

Kelix-SFT is designed to be driven by the hidden states of the Kelix unified LLM. A typical generation pipeline is:

1. Feed a text prompt (and optional images) into the **Kelix LLM**, which autoregressively produces last hidden states `{h_*}` for the image blocks.
2. Use `{h_*}` as the semantic condition (y-embedder input) for Kelix-SFT.
3. Run flow-matching denoising in the DC-AE latent space (32×32) and decode with DC-AE to obtain a 1024×1024 image.

> ⚠️ This checkpoint is a **component** of the Kelix pipeline, not a standalone text-to-image model. To generate images end-to-end you also need the Kelix unified LLM and the frozen DC-AE-F32C32 VAE. For broader scenario generalization, see the pretraining-stage checkpoint [`OpenOneRec/Kelix-DiT`](https://huggingface.co/OpenOneRec/Kelix-DiT).

## Key Results

Kelix (8B, with the SFT-stage de-tokenizer) image-generation results:

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

```bibtex
@techreport{kelix2026,
  title   = {Kelix Technique Report: Closing the Understanding Gap of Discrete Tokens in Unified Multimodal Models},
  author  = {OneRec Team},
  year    = {2026},
  url     = {https://arxiv.org/pdf/2602.09843}
}
```

## License

Please contact the OneRec Team for the license of the Kelix series. The base SANA-DiT and DC-AE components are subject to their respective original licenses.

# Kelix

[Paper](https://arxiv.org/pdf/2602.09843) | [Kelix-DiT](https://huggingface.co/OpenOneRec/Kelix-DiT) | [Kelix-SFT](https://huggingface.co/OpenOneRec/Kelix-SFT)

<p align="center">
  <img src="assets/fig3.png" alt="Kelix training pipeline: Kelix-Tok, Unified LLM, and Image DiT" width="90%">
</p>

<p align="center"><b>Figure 1:</b> The auto-regressive training workflow of Kelix, including the Kelix Tokenizer, the Unified LLM, and the Image DiT de-tokenizer.</p>

## Introduction

**Kelix** is a fully discrete autoregressive unified multimodal model by the OneRec Team. It closes the long-standing understanding gap between discrete and continuous visual representations, unifying **multimodal understanding** and **image generation** under a single autoregressive objective.

The name *Kelix* is a portmanteau of **K**uaishou and h**elix** — just as the DNA double helix encodes the full complexity of life using only four discrete nucleotide bases, Kelix encodes rich visual semantics using discrete tokens.

Kelix is built on a modular *Tokenizer → LLM → Detokenizer* pipeline:

- **Kelix-Tok** — a multi-token vision tokenizer that decomposes each patch embedding into `N` parallel discrete codes (`N=8`, total codebook size `S=65,536`), expanding the coding capacity exponentially while keeping the LLM context length unchanged via sum pooling on the encoder side. NaViT encoder initialized from Keye-VL 1.5; codebooks K-means initialized and trained with SimVQ.
- **Kelix-LLM** — a unified Qwen3-8B backbone trained with a **Next-Block Prediction (NBP)** paradigm (text block size 2, visual block size `N+1=9`), with the vocabulary expanded by 65,536 visual entries.
- **Kelix-DiT** — a diffusion-based image de-tokenizer (SANA-DiT based, flow-matching, DC-AE-F32C32 VAE) that turns the LLM's hidden states into 1024×1024 images.

Kelix achieves state-of-the-art results among comparable-scale unified models on both understanding and generation benchmarks; notably, it reaches **86.7 on OCRBench**, matching continuous-feature VLMs and surpassing the previous best discrete model by **+23%**.

## Model Zoo

| Model | Description | HuggingFace |
|---|---|---|
| **Kelix-DiT** | Pretraining-stage diffusion-based image de-tokenizer. Renders 1024×1024 images from Kelix-LLM hidden states. | [`OpenOneRec/Kelix-DiT`](https://huggingface.co/OpenOneRec/Kelix-DiT) |
| **Kelix-SFT** | Complete end-to-end release at the SFT stage — bundles Kelix-Tok + Kelix-LLM + Kelix-DiT (SFT). Use this for both understanding and generation. | [`OpenOneRec/Kelix-SFT`](https://huggingface.co/OpenOneRec/Kelix-SFT) |
| *(frozen, not included)* | DC-AE-F32C32 latent VAE (32× spatial downsampling → 32×32 latent) — from [`Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/vae`](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/tree/main/vae). | — |

## Quick Start

### Image understanding + image-token generation

```bash
python examples/sana/ar_dit/opensource/demo_kelix.py
```

Loads the Kelix unified model (Kelix-Tok + Kelix-LLM) and runs:
1. **Image understanding** — feed an image + question, get a text answer.
2. **Image-token generation** — give a text prompt, generate discrete visual tokens.

### Text-to-image generation

```bash
python examples/sana/ar_dit/opensource/demo_kelix_t2i.py \
    --prompt "Generate an image of a cute cat." \
    --output ./kelix_t2i_demo.png
```

Loads Kelix AR model + Kelix-DiT + DC-AE VAE, then:
1. AR model generates image tokens + last hidden states.
2. DiT flow-matching sampling in VAE latent space (32×32).
3. VAE decode → 1024×1024 PIL image.

Override paths via env vars: `KELIX_DIR`, `DIT_DIR`, `VAE_DIR`, `MODEL_CONFIG_OVERRIDES`.

## Key Results

**Image understanding**

| Benchmark | SEED-Bench | RealWorldQA | MMBench-EN | AI2D | MMMU | MathVista | ChartQA | TextVQA | OCRBench |
|---|---|---|---|---|---|---|---|---|---|
| Score | 76.0 | 72.1 | 80.2 | 82.4 | 54.1 | 76.5 | 83.0 | 81.4 | **86.7** |

**Image generation**

| Benchmark | GenEval (Overall) | WISE (Overall) | DPG-Bench (Overall) |
|---|---|---|---|
| Score | **87.6** | **57.0** | **85.5** |

Highlights (see the [technical report](https://arxiv.org/pdf/2602.09843) for full tables):
- **OCRBench 86.7** — matching Qwen2.5-VL-7B (86.4), **+23%** over the previous best discrete unified model X-Omni (70.4).
- **GenEval 87.6** — SOTA among discrete-tokenization unified models, **+0.6** over the 27B Qwen-Image.
- **WISE 57.0** — beats all continuous-tokenization unified models and larger dedicated T2I models.
- **DPG-Bench 85.5** — competitive with X-Omni (7B, 87.7) **without** reinforcement learning.

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

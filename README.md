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

## Installation

```bash
pip install -r requirements.txt
```

This installs `torch`, `transformers`, `diffusers`, `accelerate`, `pillow`, `numpy`, and `keye_vl_utils`. Run scripts from the repo root so the `muse/` and `recipes/` packages are importable (the demo scripts add their own dir to `sys.path` for `kelix_utils`).

## Model Zoo

| Model | Description | HuggingFace |
|---|---|---|
| **Kelix-DiT** | Pretraining-stage diffusion-based image de-tokenizer. Renders 1024×1024 images from Kelix-LLM hidden states. | [`OpenOneRec/Kelix-DiT`](https://huggingface.co/OpenOneRec/Kelix-DiT) |
| **Kelix-SFT** | Complete end-to-end release at the SFT stage — bundles Kelix-Tok + Kelix-LLM + Kelix-DiT (SFT). Use this for both understanding and generation. | [`OpenOneRec/Kelix-SFT`](https://huggingface.co/OpenOneRec/Kelix-SFT) |
| *(frozen, not included)* | DC-AE-F32C32 latent VAE (32× spatial downsampling → 32×32 latent) — from [`Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/vae`](https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers/tree/main/vae). | — |

## Quick Start

### Image understanding + image-token generation

```bash
python demo/demo_kelix.py
```

Loads the Kelix unified model (Kelix-Tok + Kelix-LLM) and runs:
1. **Image understanding** — feed an image + question, get a text answer.
2. **Image-token generation** — give a text prompt, generate discrete visual tokens.

### Text-to-image generation

```bash
python demo/demo_kelix_t2i.py \
    --prompt "Generate an image of a cute cat." \
    --output ./kelix_t2i_demo.png
```

Loads Kelix AR model + Kelix-DiT + DC-AE VAE, then:
1. AR model generates image tokens + last hidden states.
2. DiT flow-matching sampling in VAE latent space (32×32).
3. VAE decode → 1024×1024 PIL image.

Override model paths via env vars: `KELIX_DIR`, `DIT_DIR`, `VAE_DIR`, `MODEL_CONFIG_OVERRIDES`.

## Training

Example training scripts (OpenMPI / `mpirun` based) are provided:

- **Kelix-LLM (AR) training** — `examples/keye_ar/train_scripts/run_train_demo.sh` with config `examples/keye_ar/train_scripts/run_train_demo.json`.
- **Kelix-DiT training** — `examples/sana/ar_dit/demo_script/train_sft_sana.sh` with config `examples/sana/ar_dit/demo_script/train_sft_sana.json`.

Each script reads a JSON dataset config (`--dataset-config`) whose `sources` field points to your data index. Edit the paths / set the `MODEL_DIR`, `KEYE_AR_DIR`, `VAE_DIR`, `DATASET_CONFIG` environment variables to point to your model checkpoints and data before running.

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
- **DPG-Bench 85.5** — competitive with X-Omni (7B, 87.7) **without** reinforced learning.

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

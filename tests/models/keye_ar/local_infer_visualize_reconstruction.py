"""
Local single-GPU inference demo: visualize DiT reconstructions using Keye AR

Simplified version without distributed setup - runs on a single GPU for local debugging.

Usage:
    python tests/models/keye_ar/local_infer_visualize_reconstruction.py \
        --model-dir /path/to/model \
        --vae-dir /path/to/vae \
        --keye-ar-dir /path/to/keye_ar \
        --output-dir ./vis_output
"""

import argparse
import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoProcessor
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

# Import DCP converter
from muse.tools.dcp2torch import convert as dcp_to_torch_convert

# Reuse helpers from training recipe
from recipes.sana import train_sana_ar_dit as train_rec
from muse.config import load_config
from muse.models import get_model_class
from muse.models.keye_ar import KeyeARModel
from muse.utils.common import parse_config_overrides


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--dcp-ckpt-dir", type=str, default=None)
    parser.add_argument("--dcp-tag", type=str, default=None)
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--model-config-overrides", type=str, nargs="*", default=[])
    parser.add_argument("--vae-dir", type=str, required=True)
    parser.add_argument("--keye-ar-dir", type=str, required=True)
    parser.add_argument("--dataset-config", type=str,
                        default="examples/sana/ar_dit/run_ar_dit_lzx_4096_v2_1024im_multiscale.json")
    parser.add_argument("--parquet-path", type=str,
                        default="/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet")
    parser.add_argument("--output-dir", type=str, default="./vis_output")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "fl
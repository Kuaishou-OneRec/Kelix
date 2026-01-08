"""
Local single-GPU inference demo: visualize DiT reconstructions using Keye AR

This is a simplified version of infer_visualize_reconstruction.py for local development/debugging.
No distributed setup required - runs on a single GPU.

Usage example:
    python tests/models/keye_ar/local_infer_visualize_reconstruction.py \
        --model-dir /path/to/model \
        --vae-dir /path/to/vae \
        --keye-ar-dir /path/to/keye_ar \
        --dataset-config examples/sana/ar_dit/dataset.json \
        --parquet-path /path/to/vis_data.parquet \
        --output-dir ./vis_output --num-images 8
"""

import argparse
import os
import json
import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoProcessor

# Import DCP to torch converter
from muse.tools.dcp2torch import convert as dcp_to_torch_convert

# Reuse helpers from the training recipe
from recipes.sana import train_sana_ar_dit as train_rec
from muse.config import load_config
from muse.models import get_model_class
from muse.models.keye_ar import KeyeARModel
from muse.utils.common import parse_config_overrides


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing pretrained model or checkpoint")
    parser.add_argument("--dcp-ckpt-dir", type=str, default=None,
                        help="CKPT directory for DCP checkpoint conversion (required if --dcp-tag is used)")
    parser.add_argument("--dcp-tag", type=str, default=None,
                        help="Tag for DCP checkpoint (e.g., global_step8000)
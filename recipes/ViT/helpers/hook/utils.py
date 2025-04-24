import io
import os
import torch
import json
import uuid
import torch
import decord
import base64
import logging
import traceback
import numpy as np
import pandas as pd
from copy import deepcopy
import pyarrow as pa
import torch.nn as nn
import os.path as osp
from PIL import Image
import math
import multiprocessing
from io import BytesIO
import pyarrow.parquet as pq
from recipes.ViT.helpers.hook import build_hook
from typing import Union, Iterable, Optional, List, Dict, Tuple, Any
logger = logging.getLogger(__name__)


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize_image(image, multiple_of):

    if isinstance(image, Image.Image):
        width, height = image.size
        neww = round_by_factor(width, multiple_of)
        newh = round_by_factor(height, multiple_of)
        image = image.resize((neww, newh))
        return image, (neww // multiple_of) * (newh // multiple_of)
    raise NotImplementedError


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def process_image(image, **kwargs):
    if len(kwargs) == 0:
        return image
    raise NotImplementedError


def process_image_block(image_block, min_token_per_image=1, max_token_per_image=-1, patch_size=14, drop_ratio=0.0, force_size=None, multiple_of=None):
    multiple_of = multiple_of or patch_size
    assert multiple_of % patch_size == 0

    if force_size is not None:
        assert isinstance(force_size, (int, tuple))
        if isinstance(force_size, int):
            force_size = (force_size, force_size)
        assert len(force_size) == 2

    if isinstance(image_block, str):
        if len(image_block) > 200:
            base64_data = image_block
            data = base64.b64decode(base64_data)
            image = Image.open(BytesIO(data))
        else:
            image = Image.open(image_block)
        
        if force_size:
            image = image.resize(force_size, Image.Resampling.LANCZOS)
            return image

        if max_token_per_image == -1:
            return image

        width, height = image.size
        new_height, new_width = smart_resize(
            height, 
            width, 
            factor=patch_size, 
            min_pixels=multiple_of * multiple_of * min_token_per_image,
            max_pixels=multiple_of * multiple_of * max_token_per_image
        )

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        num_token = ((new_height // multiple_of) * (new_width // multiple_of))

        num_token_after_drop = max(int(num_token * (1. - drop_ratio)), 1)

        if num_token_after_drop > max_token_per_image:
            logger.error(f"Sample image (size {image.size}) num token after drop '{num_token_after_drop}' beyond max length '{max_token_per_image}'")
            return None

        return image
    elif isinstance(image_block, dict):
        image_obj = image_block.pop("image")
        image_obj = process_image_block(image_obj, max_token_per_image=max_token_per_image, patch_size=patch_size, drop_ratio=drop_ratio, force_size=force_size)
        return process_image(image_obj, **image_block)
    else:
        raise TypeError("Unsupported image_block type '{}'".format(image_block.__class__.__name__))


def process_video(video_path, nframes=10, **kwargs):
    if len(kwargs) == 0:
        vr = decord.VideoReader(video_path)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()
        video = torch.tensor(video)
        return list(video.unbind(0))
    raise NotImplementedError


def process_video_block(video_block, nframes=10):
    if isinstance(video_block, str):
        return process_video(video_block, nframes=nframes)
    elif isinstance(video_block, dict):
        video_path = video_block.pop("video")
        return process_video(video_path, nframes=nframes, **video_block)
    else:
        TypeError("Unsupported video_block type '{}'".format(video_block.__class__.__name__))


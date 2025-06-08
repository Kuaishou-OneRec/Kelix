from . import BaseHook
import torch
from einops import rearrange
import numpy as np
import logging
from PIL import Image
logger = logging.getLogger(__name__)


class VisionLengthCheckHook(BaseHook):

    def __init__(self, processor, **kwargs):
        super().__init__(processor, **kwargs)
        self.processor = processor
        self.patch_size = 14

    def __call__(self, sample, row_info_str):
        if self.truncate_text_length > 0:
            images = sample["images"]
            source = sample["source"]
            assert isinstance(images, (list, Image.Image))
            if isinstance(images, Image.Image):
                images = [images]
            
            for image in images:
                width, height = image.size
                if width
            if length > self.truncate_text_length:
                logger.warning(f"Row {row_info_str} has {length} tokens, beyond max text length {self.truncate_text_length}, skip. Data source {source}")
                return None
        return sample

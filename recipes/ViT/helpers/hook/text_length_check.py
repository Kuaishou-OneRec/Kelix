from . import BaseHook
import torch
from einops import rearrange
import numpy as np
import logging
from PIL import Image
logger = logging.getLogger(__name__)


class TextLengthCheckHook(BaseHook):

    def __init__(self, processor, **kwargs):
        super().__init__(processor, **kwargs)
        self.processor = processor
        self.truncate_text_length = kwargs.get("truncate_text_length", -1)

    def __call__(self, sample, row_info_str):
        if self.truncate_text_length > 0:
            texts = sample["text"]
            source = sample["source"]
            text_outputs = self.processor(text=texts, padding="longest", return_tensors="pt")
            input_ids = text_outputs["input_ids"]
            length = input_ids.shape[-1]
            if length > self.truncate_text_length:
                logger.warning(f"Row {row_info_str} has {length} tokens, beyond max text length {self.truncate_text_length}, skip. Data source {source}")
                return None
        return sample

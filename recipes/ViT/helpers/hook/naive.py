from . import BaseHook
import torch
from einops import rearrange
import numpy as np
from PIL import Image


class NaiveHook(BaseHook):

    def __init__(self, processor, **kwargs):
        super().__init__(processor, **kwargs)
        self.processor = processor

    def __call__(self, sample):
        return sample
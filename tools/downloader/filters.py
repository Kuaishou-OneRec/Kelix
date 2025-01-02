from typing import Dict
import pandas as pd
import numpy as np

class FilterBase(object):

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class SampleFilterBase(object):

    def __call__(self, sample: Dict[str, any]) -> bool:
        raise NotImplementedError

class ScoreFilter(FilterBase):

    def __init__(self, score_name, threshold):
        self.score_name = score_name
        self.threshold = threshold
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df[self.score_name] >= self.threshold
        return df[mask]

class CaptionLengthFilter(FilterBase):

    def __init__(self, caption_name, min_length, max_length):
        self.caption_name = caption_name
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = [
            self.min_length <= len(v) <= self.max_length
            for v in df[self.caption_name].values
        ]
        return df[mask]

class CoyoSampleFilter(SampleFilterBase):

    def __call__(self, sample: Dict[str, any]) -> bool:
        word_count = len(sample["caption"].split(" "))
        char_count = len(sample["caption"])
        caption_mask = (word_count > 2) & (char_count > 5)
        clip_score = sample["clip_similarity_vitb32"]
        clip_score_mask = clip_score >= 0.315 
        width = sample["width"]
        height = sample["height"]
        aspect_ratio = width / height if width > height else height / width
        image_size = min(width, height)
        image_mask = (image_size >= 200) & (aspect_ratio <= 3.0)
        if caption_mask & image_mask & clip_score_mask:
            return True
        else:
            return False

    
def create_filter(class_name, kwargs):
    return eval(class_name)(**kwargs)

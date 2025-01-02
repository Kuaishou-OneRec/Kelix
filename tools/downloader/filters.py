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
    
def create_filter(class_name, kwargs):
    return eval(class_name)(**kwargs)
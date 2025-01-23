from typing import Dict
from .filter import FilterBase

class ScoreFilter(FilterBase):

    def __init__(self, score_name: str, min_score: float, max_score: float, default_score: float = 0.0):
        self.score_name = score_name
        self.min_score = min_score
        self.max_score = max_score
        self.default_score = default_score
    
    def __call__(self, src: Dict[str, any]) -> bool:
        meta = self.get_metadata(src)
        score = meta.get(self.score_name, self.default_score)
        return self.min_score <= score <= self.max_score
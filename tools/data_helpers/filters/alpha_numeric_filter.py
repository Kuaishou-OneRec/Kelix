from typing import Dict
from .filter import FilterBase

class AlphaNumericFilter(FilterBase):

    def __init__(self, min_ratio: float, max_ratio: float):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def __call__(self, src: Dict[str, any]) -> bool:
        text = self.extract_all_text(src)

        alnum_count = sum(
            map(lambda char: 1 if char.isalnum() else 0, text))
        
        ratio = alnum_count / len(text) if len(text) != 0 else 0.0

        return self.min_ratio <= ratio <= self.max_ratio
    
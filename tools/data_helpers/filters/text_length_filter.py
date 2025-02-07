import json
from typing import Dict
from .filter import FilterBase

class TextLengthFilter(FilterBase):

    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length
    
    def __call__(self, src: Dict[str, any]) -> bool:
        text = self.extract_all_text(src)
        return self.min_length <= len(text) <= self.max_length

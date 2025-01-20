import numpy as np
from typing import Dict, List, Optional
from .converter import ConverterBase

class ResampleConverter(ConverterBase):

    def __init__(
        self,
        sample_config: Dict[str, float],
        default_ratio: float = 1.0,
    ):
        self.sample_config = sample_config
        self.default_ratio = default_ratio
    
    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        source = src['source']
        sample_ratio = self.sample_config.get(source, self.default_ratio)
        if np.random.rand() < sample_ratio:
            return src
        else:
            return None
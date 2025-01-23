import json
import base64
import traceback
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Dict
from .filter import FilterBase

class ImageSizeFilter(FilterBase):

    def __init__(
        self, 
        min_pixel: int, 
        max_pixel: int, 
        min_aspect_ratio: float,
        max_aspect_ratio: float):
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
    
    def __call__(self, src: Dict[str, any]) -> bool:
        images = json.loads(src['images'])
        all_valid = all([
            self.is_image_valid(x) for x in images.values()
        ])
        return all_valid

    def is_image_valid(self, img_b64):
        try:
            img_bytes = base64.b64decode(img_b64)
            img_bytes_stream = BytesIO(img_bytes)
            image = Image.open(img_bytes_stream)
            pixel_num = np.prod(image.size)
            aspect_ratio = image.size[1] / image.size[0]
            return (
                (self.min_pixel <= pixel_num <= self.max_pixel) and 
                (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio)
            )
        except Exception as e:
            print(traceback.format_exc())
            return False
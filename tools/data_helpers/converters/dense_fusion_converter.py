import os
import json
import zipfile
import numpy as np
from functools import lru_cache
from typing import Dict, List, Sequence, Optional
from .converter import (
    ConverterBase,
    render_image_text
)

class DenseFusionConverter(ConverterBase):

    def __init__(
        self,
        image_dir: str,
        source: str,
        prompts: Sequence[str],
    ):
        self.image_dir = image_dir
        self.source = source
        self.prompts = list(prompts)
    
    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        image_path = src['image_path']
        caption = src['caption']
        prompt = np.random.choice(self.prompts)

        img_name = os.path.basename(image_path)
        zip_path = os.path.join(self.image_dir, os.path.dirname(image_path) + ".zip")
        image_bytes = self.open_zip_file(zip_path).read(img_name)

        images = {"0.jpg": image_bytes}

        messages = [
            {
                "role": "user",
                "content": render_image_text(images) + [
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption}
                ]
            }
        ]
        sample = {}
        sample.update(images)
        meta = {
            "source": self.source,
            "messages": messages,
        }
        sample["json"] = json.dumps(meta)
        return sample

    @lru_cache(100)
    def open_zip_file(self, filename):
        f = zipfile.ZipFile(filename, "r")
        return f

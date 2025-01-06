import os
import json
from typing import Dict, List, Sequence, Optional

from .converter import (
    ConverterBase
)

class LlavaCC3MPretrainConverter(ConverterBase):

    def __init__(self, image_dir: str, source: str):
        self.image_dir = image_dir
        self.source = source
    
    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        image_name = src['image']
        image_path = os.path.join(self.image_dir, image_name)
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        def render_user(text):
            content = []
            subs = text.split("\n")
            for sub in subs:
                if sub == "<image>":
                    content.append({
                        "type": "image", "image": "0.jpg",
                    })
                else:
                    content.append({
                        "type": "text", "text": sub
                    })
            return content
        
        messages = [
            {
                "role": "user",
                "content": render_user(src["conversations"][0]["value"]),
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": src["conversations"][1]["value"]}
                ]
            }
        ]

        sample = {"0.jpg": image_bytes}
        meta = {
            "source": self.source,
            "messages": messages
        }
        sample["json"] = json.dumps(meta)
        return sample
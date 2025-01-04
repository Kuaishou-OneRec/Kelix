import os
import json
from typing import Dict, List, Optional
from .converter import (
    Converter,
    render_image_text
)
import hashlib

class TheCaulDronConverter(Converter):

    def __init__(self, image_dir: Optional[str] = None):
        self.image_dir = image_dir

    def __call__(self, src: Dict[str, any]) -> Dict[str, any]:
        sample = dict()
        images = {}
        for i in range(len(src['images'])):
            assert src['images'][i]['bytes'] is not None, src['images'][i]
            img_bytes = src['images'][i]['bytes']
            if self.image_dir is not None:
                md5 = hashlib.md5(img_bytes).hexdigest()
                filename = os.path.join(self.image_dir, f"{md5}.jpg")
                if not os.path.exists(filename):
                    with open(filename, "wb") as f:
                        f.write(img_bytes)
                images[filename] = img_bytes
            else:
                images[f"{i}.jpg"] = img_bytes
        if self.image_dir is None:
            sample.update(images)
        chat = src['text']
        meta = {"source": chat["source"]}
        messages = []
        messages.append({
            "role": "user",
            "content": render_image_text(images) + [
                {"type": "text", "text": chat["user"]}
            ]
        })
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": chat["assistant"]}
            ]
        })
        meta['message'] = messages
        sample['json'] = json.dumps(meta)
        return sample
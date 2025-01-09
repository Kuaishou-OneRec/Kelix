import json
from typing import Dict, List, Optional
from .converter import (
    ConverterBase,
    render_image_text
)

class DocmatrixConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> List[Dict[str, any]]:
        samples = []
        images = {}
        for i in range(len(src['images'])):
            images[f"{i}.jpg"] = src['images'][i]['bytes']

        images_content = render_image_text(images)

        results = []
        for text in src['texts']:
            sample = dict()
            meta = {"source": text["source"]}
            messages = []
            messages.append({
                "role": "user",
                "content": images_content + [
                    {"type": "text", "text": text["user"]}
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text["assistant"]}
                ]
            })
            meta["messages"] = messages
            sample["json"] = json.dumps(meta)
            sample.update(images)
            results.append(sample)
        return results

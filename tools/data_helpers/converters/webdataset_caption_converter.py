import json
import uuid
import base64
from typing import Dict, List, Sequence, Optional
from .converter import (
    ConverterBase
)


class WebDatasetCaptionConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        metadata = json.loads(src['json'])
        caption = src['txt'].decode('utf-8')
        images = {
            "0.jpg": base64.b64encode(src['jpg']).decode('ascii')
        }

        segments = [
            {"type": "text", "text": f"<{self.source}>"},
            {"type": "image", "image": "0.jpg"},
            {"type": "text", "text": caption},
        ]
        result = {
            "images": json.dumps(images),
            "videos": "null",
            "source": self.source,
            "messages": "null",
            "segments": json.dumps(segments),
            "metadata": json.dumps(metadata),
            "uuid": str(uuid.uuid1()),
        }
        return result
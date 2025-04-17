import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase

class OpenImagesCaptionConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        text = src['caption']
        image_bytes = src['image']['bytes']
        image_path = src['image']['path']

        images = {image_path: base64.b64encode(image_bytes).decode("ascii")}

        messages = None
        
        segments = []

        segments.append(
            {
                "type": "image",
                "image": image_path
            }
        )
        segments.append(
            {
                "type": "text",
                "text": text
            }
        )


        metadata = None
        result = {
            "images": json.dumps(images),
            "videos": json.dumps(None),
            "source": self.source,
            "messages": json.dumps(messages),
            "segments": json.dumps(segments),
            "metadata": json.dumps(metadata),
            "uuid": str(uuid.uuid1()),
        }
        return result
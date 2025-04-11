import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
import os.path as osp


class SFTDetailCaptionConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        messages = json.loads(src['messages'])
        images = json.loads(src['images'])
        videos = []
        metadata = src

        result = {
            "images": json.dumps(images),
            "videos": json.dumps(videos),
            "source": self.source,
            "messages": json.dumps(messages),
            "segments": json.dumps(None),
            "metadata": json.dumps(metadata),
            "uuid": str(uuid.uuid1()),
        }
        return result
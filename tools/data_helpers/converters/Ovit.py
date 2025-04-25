import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
import base64
import requests
import os


import requests
import base64


class OvitConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        images = json.loads(src['images'])
        image = list(images.keys())[0]
        image = images[image]
        #messages = json.loads(src['messages'])
        #text = messages[-1]["content"][0]["text"]
        segments = json.loads(src['segments'])
        text = segments[1]["text"]
        sample = {
                "source": self.source,
                "task": "vit",
                "images": json.dumps([image]),
                "videos": json.dumps(list()),
                "text": text,
                "metadata": json.dumps(None),
                "uuid": str(uuid.uuid1()),
            }
        return sample
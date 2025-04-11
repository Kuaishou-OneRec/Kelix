import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
import os.path as osp
import json
import hashlib
import uuid
import base64
import os

class DetailCaption_2_5_Converter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        prompt = src['prompt']
        answer = src['response']
        images_path = json.loads(src['images'])

        images = {}
        for item in images_path:
            with open(item, 'rb') as img_file:
                img_data = img_file.read()
                image_data = base64.b64encode(img_data).decode('ascii')
                image_name = os.path.basename(item)
                images[image_name] = image_data
        
        segments = None
        messages = []
        user_content = [{"type": "image", "image": os.path.basename(item)} for item in images_path]
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        videos = []
        metadata = src
        result = {
            "images": json.dumps(images),
            "videos": json.dumps(videos),
            "source": self.source,
            "messages": json.dumps(messages),
            "segments": json.dumps(segments),
            "metadata": json.dumps(metadata),
            "uuid": str(uuid.uuid1()),
        }
        return result

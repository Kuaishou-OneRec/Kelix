import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
import base64
import requests
import os



def image_url_to_base64(image_url):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image_bytes = response.content
        base64_data = base64.b64encode(image_bytes).decode("ascii") # 转换为 Base64 字符串
        return base64_data
    except Exception as e:
        print(f"Error: {e}")
        return None

def image_key_to_base64(image_key):
    image_path = f"/llm_reco/luoxinchen/dataset/GRIT/webdataset/{image_key[:-4]}"#use key's pre 5 char to find the image
    image_path = image_path + f"{image_key}.jpg"
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
            base64_data = base64.b64encode(image_bytes).decode("ascii") # 转换为 Base64 字符串
            return base64_data
    except Exception as e:
        print(f"Error: {e}")
        return None

class OpenImagesCaptionConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        text = src['caption']
        key = src['key']
        image_bytes = image_key_to_base64(key)
        if image_bytes is None:
            return None
        
        messages = None

        images = {"0.jpg": image_bytes}
        
        segments = []

        segments.append(
            {
                "type": "image",
                "image": '0.jpg'
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
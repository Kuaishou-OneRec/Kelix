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

def image_url_to_base64(image_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.gettyimages.com/'
        }
        response = requests.get(image_url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()
        
        # Check if content is actually an image
        if 'image' not in response.headers.get('Content-Type', ''):
            print("Error: URL does not point to an image")
            return None
            
        image_bytes = response.content
        base64_data = base64.b64encode(image_bytes).decode("ascii")
        return base64_data
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None


def image_key_to_base64(temp_path):
    image_path = f"/llm_reco/luoxinchen/dataset/Detailed_Caption/Detailed_Caption/densecap_data/{temp_path}.jpg"#use key's pre 5 char to find the image
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
        image = src['img_id']
        image_bytes = image_key_to_base64(image)
        if image_bytes is None:
            return None
        messages = None
        images = {"0.jpg": image_bytes}
        
        segments = []
        text = src['detailed_caption']
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
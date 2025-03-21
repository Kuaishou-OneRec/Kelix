import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient
import os


class MultiOCRConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        frames_info = src['frames']
        images_list = [item['imageUrl'] for item in frames_info.values()]
        images = {}
        for item in images_list:
            with open(item, 'rb') as img_file:
                img_data = img_file.read()
                image_data = base64.b64encode(img_data).decode('ascii')
                image_name = os.path.basename(item)
                images[image_name] = image_data
        
        segments = None
        messages = []

        for value in frames_info.values():
            messages.append({
                "role": "user",
                "content":[
                {"type": "image", "image": os.path.basename(value['imageUrl'])},
                {"type": "text", "text": "请给出这张图片中的ocr文本信息。"},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": '这张图片中的ocr信息是：' + value['text'],
            })

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

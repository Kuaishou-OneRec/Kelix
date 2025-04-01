import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient


class ASRTextConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        asr_content = src['caption']
        images = {}
        videos = []
        messages = None
        text = ""
        for item in asr_content.split('\n\n'):
            if "无对话内容" in item:
                continue
            else:
                text += "\n" + item
        if text == "":
            return None
        
        segments = [
            {"type": "text", "text": text},
        ]
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

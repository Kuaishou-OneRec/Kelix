import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient


class BlobstoreCaptionConverter(ConverterBase):

    def __init__(
        self,
        bucket_name: str,
        key_col: str,
        caption_col: str,
        source: str,
        caller: str = "recovlm_blobstore_downloader",
        md5_col: Optional[str] = None,
    ):
        self.bucket_name = bucket_name
        self.key_col = key_col
        self.caption_col = caption_col
        self.source = source
        self.md5_col = md5_col
        self.client = BlobStoreClient(caller=caller)

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        if self.md5_col is not None:
            src['md5'] = hashlib.md5(src[self.md5_col].encode()).hexdigest()

        key = src[self.key_col]
        img_bytes = self.client.get_object(self.bucket_name, key)
        if img_bytes is None:
            return None

        images = {
            "0.jpg": base64.b64encode(img_bytes).decode("ascii")
        } 
        videos = []
        messages = "null"

        segments = [
            {"type": "text", "text": f"<{self.source}>"},
            {"type": "image", "image": "0.jpg"},
            {"type": "text", "text": src[self.caption_col]},
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


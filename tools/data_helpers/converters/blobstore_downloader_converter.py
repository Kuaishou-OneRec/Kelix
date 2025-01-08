import json
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient

class BlobstoreDownloaderConverter(ConverterBase):

    def __init__(
        self,
        bucket_name: str,
        key_name: str,
        text_name: str,
        caller: str = "blobstore_downloader_converter"
    ):
        self.bucket_name = bucket_name
        self.key_name = key_name
        self.text_name = text_name
        self.client = BlobStoreClient(caller=caller)
    
    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        key = src[self.key_name]
        img_bytes = self.client.get_object(self.bucket_name, key)
        if img_bytes is None:
            return None
        else:
            sample = {
                "json": json.dumps(src),
                "jpg": img_bytes,
                "text": src[self.text_name],
            }
            return sample

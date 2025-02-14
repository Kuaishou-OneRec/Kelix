import os
import json
import argparse
import subprocess
import tempfile
import numpy as np
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient

class KwaiVideoCaptionConverter(ConverterBase, KwaiVideoDownloader):

    def __init__(
        self,
        prompts,
        source,
        **kwargs
    ):
        KwaiVideoDownloader.__init__(self, **kwargs)
        self.prompts = prompts
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        photo_id = src['photo_id']
        filename = self.prepare_video(photo_id)
        if filename is not None:
            prompt = np.random.choice(self.prompts)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": filename
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": src['caption']
                        }
                    ]
                }
            ]
            meta = {
                "source": self.source,
                "messages": messages,
            }
            print("meta", meta)
            return {
                "json": json.dumps(meta)
            }
        else:
            return None
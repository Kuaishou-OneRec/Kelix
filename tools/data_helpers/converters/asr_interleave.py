import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient
import os
import os.path as osp

class ASRInterConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        asr_content = src['caption']
        video_path = src['audio_path']
        pid_str = os.path.basename(video_path).split('.')[0]
        n_frames = len(asr_content)

        images = {}
        for i in range(n_frames):
            img_pth = osp.join('/llm_reco/chuchenglong/R3/get_asr_images/images1', pid_str, "{}.jpg".format(chr(ord('a') + i)))
            with open(img_pth, 'rb') as img_file:
                img_data = img_file.read()
                image_data = base64.b64encode(img_data).decode('ascii')
                image_name = os.path.basename(img_pth)
                images[image_name] = image_data
        videos = []
        messages = None
        
        segments = []

        for i in range(n_frames):
            segments.append(
                {
                    "type": "image",
                    "image": "{}.jpg".format(chr(ord('a') + i))
                }
            )
            segments.append(
                {
                    "type": "text",
                    "image": asr_content[i]
                }
            )


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

import os
import json
import argparse
import subprocess
import tempfile
import numpy as np
from typing import Dict, Optional
from .converter import ConverterBase
from recovlm.utils.blobstore_client import BlobStoreClient

class KwaiVideoDownloader(object):

    def __init__(self, video_dir: str, ffmpeg_args: str, caller: str = "recovlm_kwai_video_downloader"):
        self.video_dir = video_dir
        self.ffmpeg_args = list(ffmpeg_args.split(" "))
        self.client = BlobStoreClient(caller=caller)
    
    def process_video(self, input_bytes, output_file):
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_input_file:
            temp_input_file.write(input_bytes)
            temp_input_file_path = temp_input_file.name
            process = subprocess.Popen(
                [
                    'ffmpeg',
                    '-i', temp_input_file_path,
                    *self.ffmpeg_args,
                    output_file
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"ffmpeg error: {stderr.decode('utf-8')}")
                return False
            else:
                return True
    
    def prepare_video(self, photo_id) -> bool:
        video_bytes = self.client.get_video(photo_id)
        if video_bytes is None:
            return None
        output_file = os.path.join(self.video_dir, f"{photo_id}.mp4")
        valid = False
        if not os.path.exists(output_file):
            valid = self.process_video(video_bytes, output_file)
        else:
            valid = True
        if valid:
            return output_file
        else:
            return None

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


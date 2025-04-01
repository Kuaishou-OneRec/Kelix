import json
import hashlib
import uuid
import base64
from typing import Dict, Optional
from .converter import ConverterBase
import os.path as osp


class VideoDetailConverter(ConverterBase):

    def __init__(
        self,
        source: str,
    ):
        self.source = source

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:
        # pid                                            158519361166
        # text      这个视频的类型和风格特点可以归纳如下：\n\n### 类型和风格特点：\n1. **搞笑/幽...
        # prompt                       归纳这个视频的类型和风格特点，并总结其主要内容和传达的信息。
        pid_str = str(src['pid'])
        prompt = src['prompt']
        answer = src['text']
        video_path = osp.join('/llm_reco/luoxinchen/dataset/InHouse/Photo/20250215/480p_60s_4fps_v2/', pid_str+'.mp4')

        images = {}
        videos = []
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                    
                ]
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        metadata = src
        result = {
            "images": json.dumps(images),
            "videos": json.dumps(videos),
            "source": self.source,
            "messages": json.dumps(messages),
            "segments": json.dumps(None),
            "metadata": json.dumps(metadata),
            "uuid": str(uuid.uuid1()),
        }
        return result
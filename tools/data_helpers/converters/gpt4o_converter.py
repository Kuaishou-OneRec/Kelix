import json
import uuid
import os
import traceback
from typing import Dict, List, Sequence, Optional, Tuple
from .converter import (
    ConverterBase
)
from tqdm import tqdm
from tools.gpt_tools.client import GPT4oClient
import numpy as np
import base64

class GPT4oConverter(ConverterBase):
    def __init__(self, prompt: str, train_prompts: List, source: str, timeout = 300):
        self.gpt4o_client = GPT4oClient(timeout=timeout)
        self.prompt = prompt
        self.train_prompts = train_prompts
        self.source = source

    def __call__(self, image_data: Tuple) -> Optional[Dict[str, any]]:
        file_path, img_name, img_bytes = image_data
        print(file_path, img_name)
        print(len(img_bytes))

        resp = self.gpt4o_client.chat(self.prompt, [img_bytes])
        if resp is not None:
            prompt = np.random.choice(self.train_prompts)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_name
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
                            "text": resp
                        }
                    ]
                }
            ]

            images = {
                img_name: base64.b64encode(img_bytes).decode('ascii')
            }

            result = {
                "images": json.dumps(images),
                "videos": json.dumps([]),
                "source": self.source,
                "messages": json.dumps(messages),
                "segments": json.dumps(None),
                "metadata": None,
                "uuid": str(uuid.uuid1()),
            }
            return result
        else:
            return None
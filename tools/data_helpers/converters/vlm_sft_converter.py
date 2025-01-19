import json
import uuid
import os
import base64
import tarfile
import traceback
from typing import Dict, List, Sequence, Optional, Tuple
from .converter import (
    ConverterBase
)
from tqdm import tqdm

class VlmSftImageConverter(ConverterBase):
    def __init__(self, data_file):
        self.image_data_dict = {}
        with open(data_file) as fp:
            for line in tqdm(fp, total=1701560):
                line = line.strip()
                if line == "":
                    continue
                node_data = json.loads(line)
                if node_data["images"] is not None and len(node_data["images"]) > 0:
                    try:
                        image_name = node_data["images"][0]
                        self.image_data_dict[image_name] = node_data
                    except:
                        print(traceback.format_exc())
                        print(line)
                        print(node_data)
                        break
        print(len(self.image_data_dict))
        for k, v in self.image_data_dict.items():
            print(k, v)
            break

    def __call__(self, image_src: Tuple) -> Optional[Dict[str, any]]:
        name, image_str = image_src
        src = self.image_data_dict[name]

        metadata = None
        source_name = f'vlm_sft-{src.get("data_source", "None")}'
        images = {
            name: image_str
        }
        messages = src.get("messages", None)
        segments = None

        result = {
            "images": json.dumps(images),
            "videos": json.dumps([]),
            "source": source_name,
            "messages": json.dumps(messages),
            "segments": json.dumps(None),
            "metadata": None,
            "uuid": str(uuid.uuid1()),
        }
        return result

class VlmSftTextConverter(ConverterBase):

    def __call__(self, src: Dict[str, any]) -> Optional[Dict[str, any]]:

        metadata = None
        source_name = f'vlm_sft-{src.get("data_source", "None")}'
        messages = src.get("messages", None)
        segments = None

        result = {
            "images": json.dumps(dict()),
            "videos": json.dumps([]),
            "source": source_name,
            "messages": json.dumps(messages),
            "segments": json.dumps(None),
            "metadata": None,
            "uuid": str(uuid.uuid1()),
        }
        return result

if __name__ == "__main__":
    converter = VlmSftTextConverter()
    from tools.data_helpers.datasets import TgzImageDataset, VlmTextJsonl
    dataset = VlmTextJsonl("/llm_reco_ssd/luoxinchen/dataset/VLM-SFT/clean_data.jsonl")
    for data in dataset:
        res = converter(data)
        print(json.dumps(res))
        break
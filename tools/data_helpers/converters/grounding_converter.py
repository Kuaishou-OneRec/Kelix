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
import traceback
import re
from io import BytesIO
from PIL import Image, ImageDraw

class GroundingConverter(ConverterBase):
    def __init__(self, trans_type = "box_trans", pos_is_relative=True, trans2segments=True):
        self.trans_type = trans_type
        self.trans_func = {
            "box_trans": self.box_trans,
            "quad_trans": self.quad_trans
        }
        self.pos_is_relative = pos_is_relative
        self.trans2segments = trans2segments

        self.miss_set = [
            "it is ..",
            "it is ##",
            "it is"
        ]
        self.miss_sent = [
            "Sorry, I might not be able to recognize it.",
            "Apologies, I may not be able to identify it.",
            "I'm sorry, but I might not recognize it.",
            "I apologize, I may not be able to recognize it.",
            "Sorry, I might not identify it.",
            "My apologies, I might not be able to recognize it.",
            "I regret to say that I might not recognize it.",
            "I'm afraid I might not be able to identify it.",
            "Sorry, I may not recognize it.",
            "I apologize, but I might not be able to identify it.",
            "I'm sorry, I may not be able to recognize it.",
            "Apologies, but I might not recognize it.",
            "Sorry, I might not be able to identify it.",
            "I regret to inform you that I might not recognize it.",
            "I'm afraid I may not be able to recognize it.",
            "My apologies, I may not identify it.",
            "Sorry, but I might not be able to recognize it.",
            "I apologize, I might not identify it.",
            "I'm sorry, but I may not recognize it.",
            "Apologies, I might not be able to recognize it.",
        ]

    def box_trans(self, prompt, image_x, image_y):
        try:
            pattern = r"<box>\[\[(\d+), (\d+), (\d+), (\d+)\]\]</box>"
            match = re.search(pattern, prompt)
            if match:
                x1, y1, x2, y2 = [int(match.group(i)) for i in range(1, 5)]
                if not self.pos_is_relative:
                    x1 = int(float(x1) / image_x * 1000)
                    y1 = int(float(y1) / image_y * 1000)
                    x2 = int(float(x2) / image_x * 1000)
                    y2 = int(float(y2) / image_y * 1000)

                replacement = f"<|box_start|>({x1}, {y1}), ({x2}, {y2})<|box_end|>"
                result = re.sub(pattern, replacement, prompt)
                return result
        except Exception as e:
            print(f"error msg: {traceback.format_exc()}, error_prompt: {prompt}")
        
        print(f"unsupport prompt: {prompt}")
        return None
    
    def quad_trans(self, prompt, image_x, image_y):
        try:
            # 使用正则表达式找到方括号中的坐标
            pattern = r'\[(.*?)\]'
            matches = re.search(pattern, prompt)
            
            if not matches:
                print(f"unsupport prompt: {prompt}")
                return None
            
            # 提取坐标字符串
            coords_str = matches.group(1)
            
            # 使用正则表达式匹配所有坐标对
            coord_pattern = r'\((\d+),\s*(\d+)\)'
            coords = re.findall(coord_pattern, coords_str)
            
            if not coords:
                print(f"unsupport prompt: {prompt}")
                return None
            
            # 转换坐标为相对位置
            relative_coords = []
            for x, y in coords:
                if not self.pos_is_relative:
                    x = int(float(x) / image_x * 1000)
                    y = int(float(y) / image_y * 1000)
                relative_coords.append((x, y))
            
            # 构建输出字符串
            output = "<|quad_start|>"
            output += ", ".join([f"({x}, {y})" for x, y in relative_coords])
            output += "<|quad_end|>"
            result = re.sub(pattern, output, prompt)
            return result
        except Exception as e:
            print(f"error: {traceback.format_exc()}, {prompt=}")
        return None
    
    def decode_base64_image(self, base64_string):
        # 解码base64字符串并返回PIL图像对象
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    
    def msg2segments(self, messages):
        segments = []
        for msg in messages:
            content = msg["content"]
            role = msg["role"]
            if isinstance(content, str):
                segments.append(
                    {"type": "text", "text": content}
                )
            else:
                for c in content:
                    if c["type"] == "text" and role == "assistant" and c["text"].strip().lower() in self.miss_set:
                        segments.append(
                            {"type": "text", "text": np.random.choice(self.miss_sent)}
                        )
                    else:
                        segments.append(c)
        return segments
    
    def __call__(self, parquet_data: Tuple) -> Optional[Dict[str, any]]:
        messages = json.loads(parquet_data["messages"])
        user_msg = messages[0]
        assistant_msg = messages[1]
        images = json.loads(parquet_data["images"])

        assert user_msg["role"] == "user" and assistant_msg["role"] == "assistant"
        assert user_msg["content"][0]["type"] == "image" and user_msg["content"][1]["type"] == "text"

        image_name = user_msg["content"][0]["image"]
        image_bytes_b64 = images[image_name]
        image = self.decode_base64_image(image_bytes_b64)
        image_x, image_y = image.size
        content_text = user_msg["content"][1]["text"]
        
        content_text = self.trans_func[self.trans_type](content_text, image_x, image_y)
        if content_text is not None:
            user_msg["content"][1]["text"] = content_text
            if self.trans2segments:
                segments = self.msg2segments(messages)
                parquet_data["messages"] = json.dumps(None)
                parquet_data["segments"] = json.dumps(segments)
            else:
                parquet_data["messages"] = json.dumps(messages)
            return parquet_data
        else:
            return None

if __name__ == "__main__":
    converter = GroundingConverter("quad_trans", pos_is_relative = False, trans2segments=True)
    from tools.data_helpers.datasets import ParquetDataset

    datapaths = [
        "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/luoxinchen/recovlm_dataset_stage1/TextOCR"
    ]

    dataset = ParquetDataset(datapaths)
    ans = 0
    for data in tqdm(dataset):
        res = converter(data)
        ans += 1
        if ans % 100 == 0:
            print(ans)
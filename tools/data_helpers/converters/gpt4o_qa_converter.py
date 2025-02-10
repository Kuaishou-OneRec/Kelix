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

class GPT4oQAConverter(ConverterBase):
    def process_messages(self, qa_msg, img_name):
        qa_msg = qa_msg.strip()
        if qa_msg.startswith("```json"):
            qa_msg = qa_msg[7:]
        if qa_msg.endswith("```"):
            qa_msg = qa_msg[:-3]
        messages = []
        try:
            data = json.loads(qa_msg)
            if not isinstance(data, list):
                raise ValueError
            for qa_pair in data:
                q = qa_pair["question"]
                a = qa_pair["answer"]
                if len(messages) == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": img_name
                                },
                                {
                                    "type": "text",
                                    "text": q
                                }
                            ]
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": q
                                }
                            ]
                        }
                    )
                
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": a
                            }
                        ]
                    }
                )
            return messages
        except:
            print(f"error: \n{qa_msg}, {traceback.format_exc()}")
            return None

    def __call__(self, parquet_data: Tuple) -> Optional[Dict[str, any]]:
        messages = json.loads(parquet_data["messages"])
        user_msg = messages[0]
        assistant_msg = messages[1]

        assert user_msg["role"] == "user" and assistant_msg["role"] == "assistant"
        img_name = user_msg["content"][0]["image"]
        qa_pair = assistant_msg["content"][0]["text"]
        qa_msg = self.process_messages(qa_pair, img_name)
        if qa_msg is not None:
            parquet_data["messages"] = json.dumps(qa_msg)
            return parquet_data
        else:
            return None

if __name__ == "__main__":
    converter = GPT4oQAConverter()
    from tools.data_helpers.datasets import ParquetDataset

    datapaths = [
        "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_xfund_json_q",
        "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhangzixing/recovlm_dataset_table/gpt4o_sroie_json_q",

    ]

    dataset = ParquetDataset(datapaths)
    ans = 0
    for data in dataset:
        res = converter(data)
        ans += 1
        if ans % 100 == 0:
            print(ans)
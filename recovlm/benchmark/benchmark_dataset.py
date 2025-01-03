"""I2I Pairwise Dataset"""
import numpy as np
import collections

from torch.utils.data import DataLoader
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import os

import json
from torch.utils.data import IterableDataset, DataLoader
from io import BytesIO
import pandas as pd
from PIL import Image
import ast

def format_text(doc, max_text_len=1000):
  items = []
  for key, text in doc.items():
    if not is_null(text):
      items.append(f"{key}: {str(text)[:max_text_len]}")
  return "\n".join(items)

class MMMUDataset(IterableDataset):
    def __init__(self,
                data_path: str,
                model_name_or_path: str,
                **kwargs):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        df = pd.DataFrame()
        for subfolder in os.listdir(data_path):
            for subfile in os.listdir(os.path.join(data_path, subfolder)):
                if subfile.startswith("validation"):
                    file_name = os.path.join(data_path, subfolder, subfile)
                    df_file = pd.read_parquet(file_name)
                    df = pd.concat([df, df_file], ignore_index=True)
        self.data = df      

    def transform(self, sample) -> dict:
        
        if sample["question_type"] == "multiple-choice":
            # multi choice
            prompt_1 = "Select the best answer to the following multiple-choice question based on the above images. Respond with only the letter ({0}) of the correct option. The question is "
            prompt_3 = "The best answer is: "
            prompt_2 = []

            options = ast.literal_eval(sample["options"])
            selects = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M"]
            selects_options = []
            if len(options) == 0:
                print("error options: ", sample["options"], sample)
            for i in range(len(options)):
                selects_options.append(selects[i])
                prompt_2.append(selects[i] + ". " + options[i])

            prompt_1 = prompt_1.format(",".join(selects_options))
            if sample["explanation"] is not None:
                content_text ="\n".join([prompt_1] + [sample["question"]] + ["The explanation of the question is " + sample["explanation"]] + ["The options are "] + prompt_2 + [prompt_3])
            else:
                content_text ="\n".join([prompt_1] + [sample["question"]] + ["The options are "] + prompt_2 + [prompt_3])
        else:
            prompt_1 = "Based on the above images, answer the following question. The question is "
            if sample["explanation"] is not None:
                content_text = "\n".join([prompt_1, sample["question"], "The explanation of the question is " + sample["explanation"], "The answer is"])
            options = [] 
        # get content
        content_image = []
        for i in range(1, 7):
            if sample[f"image_{i}"] == None:
                continue
            content_image.append({
                "type": "text",
                "text": f"This is <image {i}>."
            })
            content_image.append({
                "type": "image",
                "image": Image.open(BytesIO(sample[f"image_{i}"]["bytes"]))
            })

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": content_image + [
                    {"type": "text", "text": content_text},
                ],
            },
        ] 
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        inputs = {"prompt": text, "multi_modal_data": mm_data}
        return {"id": sample["id"], "answer": sample["answer"], "inputs": inputs, "options": options}

    def __iter__(self):
        for i in range(len(self.data)):
            cur_data = {}
            for name in list(self.data.columns):
                cur_data[name] = self.data[name].values[i]
            transform_data = self.transform(cur_data)
            if transform_data == {}:
                continue
            yield transform_data


class MMBenchENDataset(IterableDataset):
    def __init__(self,
                data_path: str,
                model_name_or_path: str,
                **kwargs):
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.prompt = 'Select the best answer to the following multiple-choice question based on the above images. Respond with only the letter (A, B, C or D) of the correct option. Context: {}\nQuestion: {}\nOptions: {}\nAnswer:'
        self.data = open(data_path).readlines()

    def transform(self, sample) -> dict:
        index = sample['index']
        image = sample['image']
        answer = sample['answer']
        hint = sample['hint'] if sample['hint'] else 'N/A'
        question = sample['question']
        multiple_choices = ['A', 'B', 'C', 'D', 'E']

        choices = sample['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
        choice_txt = '\n'.join(choice_list)

        prompt = self.prompt.format(hint, question, choice_txt)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "<img>"
                    },
                    {
                        "image": image,
                    },
                    {
                        "type": "text", 
                        "text": "</img>"
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            },
        ] 
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("text: ", text)
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        inputs = {"prompt": text, "multi_modal_data": mm_data}
        return {"id": index, "answer": answer, "inputs": inputs}

    def __iter__(self):
        for i in range(len(self.data)):
            cur_data = json.loads(self.data[i].strip())
            print("cur_data: ", cur_data)
            transform_data = self.transform(cur_data)
            yield transform_data


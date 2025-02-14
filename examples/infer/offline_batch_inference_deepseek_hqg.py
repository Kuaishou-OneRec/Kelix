from typing import Dict, Any, List
import io
import os
import json
import base64
import collections

import ray
import numpy as np

from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import cv2
from typing import Optional
# 导入时间库，用于计时
import time
# 导入文件操作库，用于删除目录
import shutil
# 导入枚举库，用于定义抽帧策略
from enum import Enum
from openai import OpenAI
import openai

ds = ray.data.read_parquet(
    ["local:///llm_reco_ssd/luoxinchen/dataset/the_cauldron/ai2d/train-00000-of-00001-2ce340398c113b79.parquet"]
)
image_folder = os.path.join(folder_path, "photo_0205_20_frame")
excel_file_path = os.path.join(folder_path, "0205_text_cmt.xlsx")

tensor_parallel_size = 4
num_instances = 6

sampling_params = SamplingParams(temperature=0.7, top_p=0.9)

# 调整图片尺寸到合适大小
def resize(image):
    """
    调整图片大小以适应指定的尺寸。
    参数:
        image (numpy.ndarray): 输入的图片，格式为numpy数组。
    返回:
        numpy.ndarray: 调整大小后的图片。
    """
    # 获取图片的原始高度和宽度
    height, width = image.shape[:2]
    # 根据图片的宽高比确定目标尺寸
    if height < width:
        target_height, target_width = 480, 640
    else:
        target_height, target_width = 640, 480
    # 如果图片尺寸已经小于或等于目标尺寸，则直接返回原图片
    if height <= target_height and width <= target_width:
        return image
    # 计算新的高度和宽度，保持图片的宽高比
    if height / target_height < width / target_width:
        new_width = target_width
        new_height = int(height * (new_width / width))
    else:
        new_height = target_height
        new_width = int(width * (new_height / height))
    # 调整图片大小
    return cv2.resize(image, (new_width, new_height))
  
# 定义方法将指定路径图片转为Base64编码
def encode_image(image_path):
  """
  将指定路径的图片进行编码
  参数:
      image_path (str): 图片文件的路径
  返回:
      str: 编码后的图片字符串
  """
  # 读取图片
  image = cv2.imread(image_path)
  # 调整图片大小
  image_resized = resize(image)
  # 将图片编码为JPEG格式
  _, encoded_image = cv2.imencode(".jpg", image_resized)
  # 将编码后的图片转换为Base64字符串
  return base64.b64encode(encoded_image).decode("utf-8")

class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model="/llm_reco_ssd/zhouyang12/models/DeepSeek-R1/",
                       tensor_parallel_size=tensor_parallel_size,
                       rope_scaling={
                            "mrope_section": [
                                16,
                                24,
                                24
                            ],
                            "rope_type": "mrope",
                            "type": "mrope"
                       },
                       limit_mm_per_prompt={
                            "image": 20,
                            "video": 10})
        self.processor = AutoProcessor.from_pretrained(
            "/llm_reco_ssd/zhouyang12/models/DeepSeek-R1/")
    
    def process(self, serialized_messages):
        messages = json.loads(serialized_messages)
        for message in messages:
            for block in message["content"]:
                if block["type"] == "image":
                    bytes = base64.b64decode(block["image"])
                    block["image"] = Image.open(io.BytesIO(bytes))
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

        return {"inputs": inputs}

    def collate(self, samples):
        batch = collections.defaultdict(list)
        for sample in samples:
            for key, item in sample.items():
                batch[key].append(item)
        return batch


    def __call__(self, batch: Dict[str, Any]) -> Dict[str, list]:
        samples = []
        # for messages in batch["messages"]:
        #     samples.append(self.process(messages))
        # 按photo_id分组
        photo_id_to_images = {}
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        for image_file in image_files:
            photo_id = image_file.split('_')[0]
            if photo_id not in photo_id_to_images:
                photo_id_to_images[photo_id] = []
            photo_id_to_images[photo_id].append(image_file)
            print("get photo_id_to_images succ, ", photo_id)

        df = pd.read_excel(excel_file_path)
        # 将 Excel 文件中的数据转换为字典
        photo_texts = {}
        for index, row in df.iterrows():
            photo_id = str(row['photo_id'])
            photo_texts[photo_id] = {
                "wenjuan_type": str(row['wenjuan_type']) if not pd.isna(row['wenjuan_type']) else "",
                "caption": str(row['caption']) if not pd.isna(row['caption']) else "",
                "user_comment": str(row['user_comment']) if not pd.isna(row['user_comment']) else "",
                "ocr": str(row['ocr']) if not pd.isna(row['ocr']) else "",
                "asr": str(row['asr']) if not pd.isna(row['asr']) else ""
            }
        for photo_id, image_files in photo_id_to_images.items():
            image_urls = []
            messages = []
            print("start process proto ", photo_id)
            
            # 获取对应的文字描述
            if photo_id in photo_texts:
                description = photo_texts[photo_id]
                context = [
                    {"type": "text", "text": "对应的视频标题是：" + description["caption"]},
                    {"type": "text", "text": "对应的视频 ocr 内容是：" + description["ocr"]},
                    {"type": "text", "text": "对应的视频 asr 识别结果是：" + description["asr"]},
                    {"type": "text", "text": "对应的短视频平台内用户评论内容是：" + description["user_comment"]}
                ]
            else:
                context = []

            # 为每张图片创建一个image_url对象
            # for image_file in image_files:
            #     image_path = os.path.join(image_folder, image_file)
            #     base64_image = encode_image(image_path)
            #     context.append({
            #         "type": "image_url",
            #         "image_url": {
            #             "url": f"data:image/jpeg;base64,{base64_image}",
            #             # "url": f"data:image/jpeg;base64,test",
            #             "detail": "high"
            #         },
            #     })
            samples.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"你是一个短视频平台内容理解专家，为了将该视频推荐给潜在感兴趣的用户，请结合短视频的视频抽帧结果、ocr 结果、asr 结果以及平台内所有用户对这个视频评论信息判断用户是否会对这个视频满意。请充分考虑评论信息，结合这些信息提炼出该视频所有的看点。",
                    },
                    *context,
                    {
                        "type": "text",
                        "text": f"请深呼吸并一步步仔细思考，请输出思考过程和结果。",
                    }
                ],
                # stream=True,
                # stream_options={"include_usage":True}
            })
        batch = self.collate(samples)
        outputs = self.llm.generate(batch["inputs"], sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]).strip())
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }

# For tensor_parallel_size > 1, we need to create placement groups for vLLM
# to use. Every actor has to have its own placement group.
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


resources_kwarg: Dict[str, Any] = {}
if tensor_parallel_size == 1:
    # For tensor_parallel_size == 1, we simply set num_gpus=1.
    resources_kwarg["num_gpus"] = 1
else:
    # Otherwise, we have to set num_gpus=0 and provide
    # a function that will create a placement group for
    # each instance.
    resources_kwarg["num_gpus"] = 0
    resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

def render_image_text(images):
    text = []
    for key in images.keys():
        text.append({
            "type": "image",
            "image": f"{key}"
        })
    return text

def parse(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = []
    images = [{"type": "image", "image": base64.b64encode(img["bytes"]).decode("utf-8")} \
        for img in raw["images"]]
    for qa in raw["texts"]:
        messages = []
        messages.append({
            "role": "user",
            "content": images + [{"type": "text", "text": qa["user"]}]
        })
        # messages.append({
        #     "role": "assistant",
        #     "content": [
        #         {"type": "text", "text": qa["assistant"]}]
        # })
        conversations.append({"messages": json.dumps(messages)})
    return conversations


# ds = ds.flat_map(parse)

# with open("results.jsonl", "w") as f:
#     batch = ds.flat_map(parse).take_batch(4)
#     print(batch)

ds = ds.flat_map(parse).map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=num_instances,
    batch_size=32,
    # Specify the batch size for inference.
    **resources_kwarg,
)

with open("results.jsonl", "w") as f:
    outputs = ds.take(limit=1000)
    for output in outputs:
        prompt = output["prompt"]
        generated_text = output["generated_text"]
        f.write(json.dumps(output, ensure_ascii=False) + "\n")

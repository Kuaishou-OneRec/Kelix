from typing import Dict, Any, List
import io
import os
import json
import base64
import collections
import sys
import yaml

import ray
import numpy as np

from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

tensor_parallel_size = 4
num_instances = 6

sampling_params = SamplingParams(temperature=0.7, top_p=0.9)

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
                            "image": 10,
                            "video": 10},
                        trust_remote_code=True)
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
        for messages in batch["messages"]:
            samples.append(self.process(messages))
        batch = self.collate(samples)
        outputs = self.llm.generate(batch["inputs"], sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
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
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_data_to_parquet.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        ds = ray.data.read_parquet(
            [config["parquet_path"]]
        )

    ds = ds.flat_map(parse).map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        batch_size=32,
        # Specify the batch size for inference.
        **resources_kwarg,
    )

    with open(config["result_path"], "w") as f:
        outputs = ds.take(limit=config["infer_num_limit"])
        for output in outputs:
            prompt = output["prompt"]
            generated_text = output["generated_text"]
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
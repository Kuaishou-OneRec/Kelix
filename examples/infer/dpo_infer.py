import io
import os
import ray
import json
import glob
import uuid
import torch
import base64
os.system("pip install decord")
import decord
import argparse
import subprocess
import numpy as np
import pandas as pd
import pyarrow as pa
import os.path as osp
from PIL import Image
import pyarrow.parquet as pq
import torch.nn.functional as F
import torch.distributed as dist
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from recovlm.utils.common import load_env, Timer
from ray.util.placement_group import placement_group
from torch.utils.data import DataLoader, IterableDataset, Dataset
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


runtime_env = {
    "env_vars": load_env()
}

ray.init(runtime_env=runtime_env)


def read_rows_in_file(fn):
    parquet_file = pq.ParquetFile(fn)
    df = parquet_file.read().to_pandas()

    return df


class VideoDataset(IterableDataset):

    def __init__(self, files, epochs=1):
        self.files = files
        self.epochs = epochs

    @staticmethod
    def decode_image(image):
        image_bytes = base64.b64decode(image)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    @staticmethod
    def check_valid_video_path(messages):
        for turn in messages:
            if turn["role"] == "user":
                for block in turn["content"]:
                    if block["type"] == "video":
                        if not isinstance(block["video"], str):
                            return False
                        path = block["video"]
                        try:
                            decord.VideoReader(path)
                        except:
                            return False
        return True

    def process_messages(self, messages, images):
        if images is None or len(images) == 0:
            if self.check_valid_video_path(messages):
                return messages
            return None
        images = {
            key: self.decode_image(images[key])
            for key in images
        }

        for turn in messages:
            if turn["role"] == "user":
                for block in turn["content"]:
                    if block["type"] == "video" and isinstance(block["video"], list):
                        for idx, image_block in enumerate(block["video"]):
                            if isinstance(image_block, dict):
                                block["video"][idx] = {
                                    "type": "image",
                                    "image": images[image_block["image"]]
                                }
                            elif isinstance(image_block, str):
                                block["video"][idx] = {
                                    "type": "image",
                                    "image": images[image_block]
                                }
                            else:
                                raise TypeError
                            block["video"][idx] = block["video"][idx]["image"]
        return messages

    def __iter__(self):
        for epoch in range(self.epochs):
            for file in self.files:
                df = read_rows_in_file(file)

                for _, row in df.iterrows():
                    rewrite_messages = self.process_messages(json.loads(row.messages), json.loads(row.images))
                    if rewrite_messages is None:
                        continue
                    yield {
                        "source": row.source,
                        "images": row.images,
                        "videos": row.videos,
                        "rewrite_messages": rewrite_messages,
                        "messages": row.messages,
                        "segments": row.segments,
                        "metadata": row.metadata,
                        "uuid": str(uuid.uuid1()),
                        "chosen": row.chosen,
                    }


def collate(samples):
    batch = dict()
    for sample in samples:
        for key in sample:
            if key not in batch:
                batch[key] = list()
            batch[key].append(sample[key])

    return batch


def build_response(response):

    block_list = list()
    for resp in response:
        block_list.append(
            json.dumps(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": resp
                        }
                    ]
                }
            )
        )
    return block_list


class Writer(object):

    def __init__(self, output_dir, prefix=""):
        self.output_dir = output_dir
        self.prefix = prefix
        self._buffer = dict()
        self._count = 0

    def flush(self, save_nrows):
        if self._count <= save_nrows:
            return
        output_dir = self.output_dir
        df = pd.DataFrame(self._buffer)
        os.makedirs(output_dir, exist_ok=True)
        save_fn = osp.join(output_dir, "{}{}.parquet".format(self.prefix, str(uuid.uuid1())))
        df.to_parquet(save_fn, index=False)
        self.clear()

    def clear(self):
        self._buffer.clear()
        self._count = 0

    def extend(self, batch):
        key_lengths = list(set([len(batch[key]) for key in batch]))
        assert len(key_lengths) == 1
        self._count += key_lengths[0]
        for key in batch:
            if key not in self._buffer:
                self._buffer[key] = list()
            self._buffer[key].extend(batch[key])


@ray.remote
class InferenceActor(object):

    def __init__(self,args, model, sampling_params, **kwargs):
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        self.rank = rank
        self.world_size = world_size

        self.args = args
        self.model_dir = args.model_dir
        self.output_dir = args.output_dir
        self.files = json.load(open(args.file_list, "r"))[rank::world_size]

        self.model = model
        self.sampling_params = sampling_params

        self.kwargs = kwargs

        self.loader = None
        self.processor = None
        self.writer = None

    def setup(self):
        self.loader = self._setup_dataloader()
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.writer = Writer(self.output_dir, prefix="rank-{}-".format(self.rank))

    def _setup_dataloader(self, args=None):
        args = args or self.args
        epochs = args.epochs
        num_workers = args.num_workers
        batch_size = args.batch_size
        dataset = VideoDataset(self.files, epochs=epochs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
        return dataloader

    def process(self, messages):
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

        return inputs

    def generate(self, input_list):
        inputs = list()
        for sample in input_list:
            inputs.append(self.process(sample))
        outputs = ray.get(self.model.generate.remote(inputs, self.sampling_params))
        return outputs

    def run(self):
        args = self.args
        save_nrows = args.save_nrows
        for batch in self.loader:
            messages = batch.pop("rewrite_messages")
            try:
                outputs = self.generate(messages)
            except:
                continue
            response_list = list()
            for output in outputs:
                generated_text = output.outputs[0].text
                response_list.append(generated_text)
            batch["rejected"] = build_response(response_list)
            self.writer.extend(batch)
            self.writer.flush(save_nrows)
            del batch
            torch.cuda.empty_cache()
        self.writer.flush(0)


class InferLLM(LLM):

    def __init__(self, *args, **kwargs):
        del os.environ["CUDA_VISIBLE_DEVICES"]
        super().__init__(*args, **kwargs)


def main(args):
    total_gpus = args.num_nodes * args.num_gpus_per_node
    world_size = total_gpus // args.tp_size

    actor_list = list()
    for rank in range(world_size):
        bundles = [{"CPU": 1, "GPU": 1} for _ in range(args.tp_size)]
        pg = placement_group(bundles, strategy="STRICT_PACK")
        ray.get(pg.ready())
        model = ray.remote(
            num_gpus=0,
            num_cpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0
            )
        )(InferLLM).remote(
            model=args.model_dir,
            enforce_eager=True,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend="ray",
            gpu_memory_utilization=0.80,
            limit_mm_per_prompt={
                "image": args.limit_mm_per_prompt,
                "video": args.limit_mm_per_prompt
            }
        )
        sampling_params = SamplingParams(
            top_p=args.top_p,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty
        )
        kwargs = {
            "rank": rank,
            "world_size": world_size
        }
        actor = InferenceActor.remote(args, model, sampling_params, **kwargs)
        actor_list.append(actor)

    ray.get([actor.setup.remote() for actor in actor_list])

    ray.get([actor.run.remote() for actor in actor_list])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--num_nodes", type=int, default=3)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--save_nrows", default=512, type=int)
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--num_gpus_per_node", type=int, default=8)
    parser.add_argument("--limit_mm_per_prompt", type=int, default=10)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--file_list", type=str, required=True, default="")
    parser.add_argument("--model_dir", type=str, default="/llm_reco/chuchenglong/DPO/output/v1-20250408-003739/checkpoint-273-merged")
    ags = parser.parse_args()
    main(ags)

"""Offline inference """
import ray
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

from ray.util.actor_pool import ActorPool
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from recovlm.utils.logger import init_logger
from recovlm.utils.common import load_env, Timer

import torch.distributed as dist

MODEL_DIR = "/llm_reco_ssd/zhouyang12/models/Qwen2.5-VL-72B-Instruct/"
TP_SIZE = 8

logger = init_logger(__name__)

runtime_env = {
  "env_vars": load_env()
}

# default processer
processor = AutoProcessor.from_pretrained(MODEL_DIR)

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "assets/53.jpeg",
                "min_pixels": 1280 * 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "识别图片中的内容，输出成LaTeX格式"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

# Create sampling parameters, the same across all ranks
sampling_params = SamplingParams(
    temperature=1, top_p=0.001, top_k=1,
    repetition_penalty=1.05,
    max_tokens=4096
)

# Use `distributed_executor_backend="external_launcher"` so that
# this llm engine/instance only creates one worker.
llm = LLM(
    model=MODEL_DIR,
    enforce_eager=True,
    tensor_parallel_size=TP_SIZE,
    distributed_executor_backend="external_launcher",
    enable_prefix_caching=True,
    gpu_memory_utilization=0.90,
    limit_mm_per_prompt={
        "image": 10,
        "video": 10
    }
)

outputs = llm.generate([
    {
        "prompt": text,
        "multi_modal_data": {
            "image": image_inputs
        }
    }
], sampling_params)

# all ranks will have the same outputs
if dist.get_rank() == 0:
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    with open("results.tex", "w") as f:
        f.write(generated_text)
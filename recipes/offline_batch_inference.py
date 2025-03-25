import ray
import os
import json
import random
import argparse
import torch
import torch.distributed as dist
from ray.util.actor_pool import ActorPool
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from torch.utils.data import IterableDataset, DataLoader

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.worker.worker import Worker

from recovlm.utils.common import load_env, Timer
from recovlm.utils.logger import init_logger
from recovlm.data.dataloaders_v2 import get_dataloader

from tqdm import tqdm

logger = init_logger(__name__)

runtime_env = {
  "env_vars": load_env()
}

ray.init(runtime_env=runtime_env)

def get_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("--model_dir", type=str, default=None,
                      help="The model directory to inference.")
  
  parser.add_argument("--output_dir", type=str, default=None,
                      help="The output directory.")
  
  parser.add_argument("--dataset_config", type=str, default=None)

  parser.add_argument("--num_workers", type=int, default=8)

  parser.add_argument("--batch_size", type=int, default=16)

  parser.add_argument("--shard_by_files", action="store_true")

  parser.add_argument("--num_gpus_per_node", type=int, default=8,
                      help="The number of nodes for TrainActor.")
  
  parser.add_argument("--num_inference_node", type=int, default=8,
                      help="The number of nodes for TrainActor.")

  parser.add_argument("--tp_size", type=int, default=8,
                      help="The number of nodes for TrainActor.")

  parser.add_argument("--num_generations", type=int, default=1)

  parser.add_argument("--max_generations_per_req", type=int, default=64)

  parser.add_argument("--temperature", type=float, default=0.6)
  
  parser.add_argument("--top_p", type=float, default=0.95)

  parser.add_argument("--top_k", type=int, default=50)
 
  parser.add_argument("--repetition_penalty", type=float, default=1.02)

  parser.add_argument("--limit_mm_per_prompt", type=int, default=10)

  parser.add_argument("--max_new_tokens", type=int, default=1024)

  return parser.parse_args()

@ray.remote
class GenerationActor:
  def __init__(self,
               args: argparse.Namespace,
               rank: int = 0,
               world_size: int = 1):
    self.args = args
    self.rank = rank
    self.world_size = world_size

  def initialize(self):
    with open(self.args.dataset_config, encoding="utf-8") as f:
      dataset_config = json.loads(f.read())
    if self.args.batch_size:
        dataset_config["batch_size"] = self.args.batch_size
    dataset = dataset_config.pop("name")

    self.dataloader = get_dataloader(
      name=dataset, rank=self.rank, world_size=self.world_size,
      **dataset_config
    )

  def set_engine(self, engine):
    self.engine = engine

  def generate(self):
    with open(f"{self.args.output_dir}/rank_{self.rank}", "w", encoding="utf-8") as out:
      for batch in tqdm(self.dataloader):
        num_generations = self.args.num_generations
        max_generations_per_req = self.args.max_generations_per_req
        all_chunks = []
        while num_generations > 0:
          if num_generations > max_generations_per_req:
            num_generations -= self.args.max_generations_per_req
            n = max_generations_per_req
          else:
            n = num_generations
            num_generations = 0
          sampling_params = SamplingParams(
            n=n,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=self.args.repetition_penalty,
            max_tokens=self.args.max_new_tokens
          )
          results = ray.get(
            self.engine.generate.remote(
              [e["vllm_inputs"] for e in batch],
              sampling_params,
              use_tqdm=True
            )
          )
          all_chunks.append(results)

        all_response = []
        for prompt_idx in range(len(batch)):
          responses = []
          for chunk in all_chunks:
            for output in chunk[prompt_idx].outputs:
              responses.append(output.text)
          all_response.append(responses)
          out.write(json.dumps({
            "annotation": batch[prompt_idx]["annotation"],
            "responses": all_response[prompt_idx],
            "source": batch[prompt_idx]["source"],
            "__key__": batch[prompt_idx]["__key__"],
            "__url__": batch[prompt_idx]["__url__"]
          }) + "\n")

class MyLLM(LLM):
  def __init__(self, *args, **kwargs):
    # a hack to make the script work.
    # stop ray from manipulating CUDA_VISIBLE_DEVICES
    # at the top-level
    del os.environ["CUDA_VISIBLE_DEVICES"]
    super().__init__(*args, **kwargs)


def main():
  args = get_arguments()
  world_size = args.num_gpus_per_node * args.num_inference_node // \
    args.tp_size

  generation_actors = []
  for rank in range(world_size):
    pg = placement_group(
      [{"GPU": 1, "CPU": 1}] * args.tp_size,
      strategy="STRICT_PACK"
    )
    ray.get(pg.ready())

    engine = ray.remote(
      num_cpus=1,
      num_gpus=0,
      scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0
      )
    )(MyLLM).remote(
      model=args.model_dir,
      enforce_eager=True,
      tensor_parallel_size=args.tp_size,
      distributed_executor_backend="ray",
      enable_prefix_caching=True,
      gpu_memory_utilization=0.80,
      limit_mm_per_prompt={
        "image": args.limit_mm_per_prompt,
        "video": args.limit_mm_per_prompt
      }
    )
    generation_actor = GenerationActor.remote(
      args=args, rank=rank, world_size=world_size)
    ray.get(generation_actor.set_engine.remote(engine))
    generation_actors.append(generation_actor)
  
  ray.get([actor.initialize.remote() for actor in generation_actors])

  ray.get([actor.generate.remote() for actor in generation_actors])

if __name__ == '__main__':
    main()
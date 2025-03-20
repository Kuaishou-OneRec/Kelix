
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from recovlm.data.datasets_v2 import ExperienceDataset, VisionPromptDataset, VllmInferenceDataset

def get_vision_prompt_dataloader(sources: str,
                                 max_length,
                                 min_visual_tokens_per_image,
                                 max_visual_tokens_per_image,
                                 shrink_ratio,
                                 max_retry,
                                 multiple_of,
                                 pretrained_model_name_or_path,
                                 num_epochs=1,
                                 seed=1024,
                                 num_workers=8,
                                 rank=0,
                                 world_size=1,
                                 batch_size=1,
                                 max_images=10,
                                 datasource_config={},
                                 **kwargs) -> DataLoader: 
  """Create dataloader for vision prompt"""
  dataset = VisionPromptDataset(
    sources=sources,
    max_length=max_length,
    min_visual_tokens_per_image=min_visual_tokens_per_image,
    max_visual_tokens_per_image=max_visual_tokens_per_image,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    shrink_ratio=shrink_ratio,
    max_retry=max_retry,
    multiple_of=multiple_of,
    num_epochs=num_epochs,
    seed=seed,
    num_workers=num_workers,
    rank=rank,
    world_size=world_size,
    batch_size=batch_size,
    max_images=max_images,
    datasource_config=datasource_config
  )
  def collate_fn(samples):
    return samples

  dataloader = StatefulDataLoader(
    dataset=dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=collate_fn)
  return dataloader

def get_experience_dataloader(
    generated: List[Dict],
    max_length: Optional[int] = None,
    batch_size: Optional[int] = 1,
    multiple_of: Optional[int] = 8,
    use_packing: Optional[bool] = True,
    num_packing_samples: Optional[int] = -1,
    pretrained_model_name_or_path: Optional[str] = None) -> DataLoader:
  """Create dataloader for GRPO experiences"""
  dataset = ExperienceDataset(
    generated=generated,
    max_length=max_length,
    multiple_of=multiple_of,
    use_packing=use_packing,
    num_packing_samples=num_packing_samples,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
  )
  if use_packing:
    collate_fn = lambda x: x[0]
  else:
    collate_fn = dataset.build_collate_fn()
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
  )

def get_vllm_inference_dataloader(sources: str,
                                  min_visual_tokens_per_image,
                                  max_visual_tokens_per_image,
                                  pretrained_model_name_or_path,
                                  seed=1024,
                                  num_workers=8,
                                  rank=0,
                                  world_size=1,
                                  batch_size=1,
                                  max_images=10):
  """Create dataloader for vision prompt"""
  dataset = VllmInferenceDataset(
    sources=sources,
    min_visual_tokens_per_image=min_visual_tokens_per_image,
    max_visual_tokens_per_image=max_visual_tokens_per_image,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    seed=seed,
    num_workers=num_workers,
    rank=rank,
    world_size=world_size,
    max_images=max_images
  )
  def collate_fn(samples):
    return samples

  dataloader = DataLoader(
    dataset=dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=num_workers,
    collate_fn=collate_fn)
  return dataloader

def get_dataloader(name: str, **kwargs):
  if name == "experience":
    return get_experience_dataloader(**kwargs)
  elif name == "vision_prompt":
    return get_vision_prompt_dataloader(**kwargs)
  elif name == "vllm_infer":
    return get_vllm_inference_dataloader(**kwargs)
  else:
    raise ValueError(f"Invalid dataloader name: {name}")

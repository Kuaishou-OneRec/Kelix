
from typing import Optional, Callable, Dict, Union, Any, List
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from recovlm.data.datasets_v2 import ExperienceDataset, VisionPromptDataset, VllmInferenceDataset, ChatCompletionVisionDatasetV2, ChatCompletionVisionV2ParquetDataset, \
  DEFAULT_SYSTEM_PROMPT

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


def get_chat_completion_vision_v2_parquet_dataloader(sources: str,
                                                   max_length: int,
                                                   min_visual_tokens_per_image: int,
                                                   max_visual_tokens_per_image: int,
                                                   base_model_dir: str,
                                                   shrink_ratio: float = 0.9,
                                                   max_retry: int = 5,
                                                   multiple_of: int = 8,
                                                   num_workers: int = 8,
                                                   rank: int = 0,
                                                   world_size: int = 1,
                                                   num_epochs: int = 1,
                                                   shuffle_seed: int = 1024,
                                                   video_nframe: int = -1,
                                                   video_fps: float = 2.0,
                                                   video_min_frames: int = 2,
                                                   video_max_frames: int = 120,
                                                   datasource_config: Dict[str, Dict[str, Any]] = {},
                                                   pretrained_model_name_or_path: str = None,
                                                   max_images: int = 10,
                                                   **kwargs
                                                   ):
    """
    创建 ChatCompletionVisionV2ParquetDataset 的 DataLoader。

    Args:
        sources (str): 数据源路径
        max_length (int): 最大序列长度
        min_visual_tokens_per_image (int): 每个图像的最小视觉token数
        max_visual_tokens_per_image (int): 每个图像的最大视觉token数
        base_model_dir (str): 基础模型目录
        shrink_ratio (float, optional): 缩小比例. Defaults to 0.9.
        max_retry (int, optional): 最大重试次数. Defaults to 5.
        multiple_of (int, optional): 序列长度需要是该值的倍数. Defaults to 8.
        num_workers (int, optional): 数据加载的工作进程数. Defaults to 8.
        num_epochs (int, optional): 训练轮数. Defaults to 1.
        shuffle_seed (int, optional): 随机种子. Defaults to 1024.
        video_nframe (int, optional): 视频帧数. Defaults to -1.
        video_fps (float, optional): 视频帧率. Defaults to 2.0.
        video_min_frames (int, optional): 最小视频帧数. Defaults to 2.
        video_max_frames (int, optional): 最大视频帧数. Defaults to 120.
        datasource_config (Dict[str, Dict[str, Any]], optional): 数据源配置. Defaults to {}.

    Returns:
        DataLoader: 返回配置好的DataLoader实例
    """
    dataset = ChatCompletionVisionV2ParquetDataset(
        sources=sources,
        num_workers=num_workers,
        num_epochs=num_epochs,
        shuffle_seed=shuffle_seed,
        max_length=max_length,
        min_visual_tokens_per_image=min_visual_tokens_per_image,
        max_visual_tokens_per_image=max_visual_tokens_per_image,
        video_nframe=video_nframe,
        video_fps=video_fps,
        video_min_frames=video_min_frames,
        video_max_frames=video_max_frames,
        base_model_dir=base_model_dir,
        shrink_ratio=shrink_ratio,
        max_retry=max_retry,
        multiple_of=multiple_of,
        datasource_config=datasource_config,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        rank=rank,
        world_size=world_size,
        max_images=max_images,
    )

    # 使用 StatefulDataLoader 以支持状态保存和恢复
    dataloader = StatefulDataLoader(
        dataset=dataset,
        shuffle=False,  # 数据集内部已经实现了shuffle
        batch_size=1,   # 由于使用了packing，batch_size固定为1
        num_workers=num_workers,
        collate_fn=lambda x: x[0]  # 简单的collate函数，因为数据集已经处理好了packing
    )
    from recovlm.utils.common import pytorch_worker_info
    print("datttttt", num_workers, pytorch_worker_info(), world_size) # 输出datttttt 4 (10, 40, 0, 1)   [rank, world_size, worker, num_workers]
    return dataloader

def get_chat_completion_vision_v2_dataloader(sources: str,
                                           max_length: int,
                                           min_visual_tokens_per_image: int,
                                           max_visual_tokens_per_image: int,
                                           base_model_dir: str,
                                           shrink_ratio: float = 0.9,
                                           max_retry: int = 5,
                                           multiple_of: int = 8,
                                           num_workers: int = 8,
                                           rank: int = 0,
                                           world_size: int = 1,
                                           seed: int = 1024,
                                           num_epochs: int = 1,
                                           video_nframe: int = -1,
                                           video_fps: float = 2.0,
                                           video_min_frames: int = 2,
                                           video_max_frames: int = 120,
                                           datasource_config: Dict[str, Dict[str, Any]] = {},
                                           pretrained_model_name_or_path: str = None,
                                           max_images: int = 10,
                                           **kwargs: Any
                                           ):
    """
    创建 ChatCompletionVisionDatasetV2 的 DataLoader。

    Args:
        sources (str): 数据源路径
        max_length (int): 最大序列长度
        min_visual_tokens_per_image (int): 每个图像的最小视觉token数
        max_visual_tokens_per_image (int): 每个图像的最大视觉token数
        base_model_dir (str): 基础模型目录
        shrink_ratio (float, optional): 缩小比例. Defaults to 0.9.
        max_retry (int, optional): 最大重试次数. Defaults to 5.
        multiple_of (int, optional): 序列长度需要是该值的倍数. Defaults to 8.
        num_workers (int, optional): 数据加载的工作进程数. Defaults to 8.
        rank (int, optional): 当前进程的rank. Defaults to 0.
        world_size (int, optional): 总进程数. Defaults to 1.
        seed (int, optional): 随机种子. Defaults to 1024.
        num_epochs (int, optional): 训练轮数. Defaults to 1.
        video_nframe (int, optional): 视频帧数. Defaults to -1.
        video_fps (float, optional): 视频帧率. Defaults to 2.0.
        video_min_frames (int, optional): 最小视频帧数. Defaults to 2.
        video_max_frames (int, optional): 最大视频帧数. Defaults to 120.
        datasource_config (Dict[str, Dict[str, Any]], optional): 数据源配置. Defaults to {}.

    Returns:
        DataLoader: 返回配置好的DataLoader实例
    """
    dataset = ChatCompletionVisionDatasetV2(
        sources=sources,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        seed=seed,
        num_epochs=num_epochs,
        max_length=max_length,
        min_visual_tokens_per_image=min_visual_tokens_per_image,
        max_visual_tokens_per_image=max_visual_tokens_per_image,
        video_nframe=video_nframe,
        video_fps=video_fps,
        video_min_frames=video_min_frames,
        video_max_frames=video_max_frames,
        base_model_dir=base_model_dir,
        shrink_ratio=shrink_ratio,
        max_retry=max_retry,
        multiple_of=multiple_of,
        datasource_config=datasource_config,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        max_images=max_images
    )

    # 使用 StatefulDataLoader 以支持状态保存和恢复
    dataloader = StatefulDataLoader(
        dataset=dataset,
        shuffle=False,  # 数据集内部已经实现了shuffle
        batch_size=1,   # 由于使用了packing，batch_size固定为1
        num_workers=num_workers,
        collate_fn=lambda x: x[0]  # 简单的collate函数，因为数据集已经处理好了packing
    )
    
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
                                  max_images=10,
                                  system_prompt = DEFAULT_SYSTEM_PROMPT):
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
    max_images=max_images,
    system_prompt=system_prompt
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
  elif name == "chat_vision_v2":
    return get_chat_completion_vision_v2_dataloader(**kwargs)
  elif name == "chat_vision_v2_parquet":
    return get_chat_completion_vision_v2_parquet_dataloader(**kwargs)
  else:
    raise ValueError(f"Invalid dataloader name: {name}")

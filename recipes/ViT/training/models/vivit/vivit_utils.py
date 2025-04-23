import av
import numpy as np
import torch
import sys
import os


# 使用相对导入
from recipes.ViT.training.models.vivit.image_processing_vivit import VivitImageProcessor
from recipes.ViT.training.models.vivit.modeling_vivit import VivitForVideoClassification
from huggingface_hub import hf_hub_download


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def read_image_pil(video,indices):
    frames = []
    for i in indices:
        frames.append(video[i])
    return np.stack([x for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    # 如果视频帧数很少，使用连续重复采样
    if seg_len <= clip_len:
        # 计算每个帧需要重复的次数
        repeat_per_frame = clip_len // seg_len
        remainder = clip_len % seg_len
        
        # 创建结果数组
        indices = []
        
        # 对每个帧进行连续重复
        for i in range(seg_len):
            # 计算当前帧需要重复的次数
            current_repeat = repeat_per_frame + (1 if i < remainder else 0)
            # 连续重复当前帧
            indices.extend([i] * current_repeat)
        
        # 转换为numpy数组
        indices = np.array(indices, dtype=np.int64)
    else:
        # 如果视频帧数足够，使用原来的采样方法
        if clip_len * frame_sample_rate < seg_len:
            converted_len = int(clip_len * frame_sample_rate)
            end_idx = np.random.randint(converted_len, seg_len)
            start_idx = end_idx - converted_len
            indices = np.linspace(start_idx, end_idx, num=clip_len)
            indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        else:
            indices = np.linspace(0, seg_len-1, num=clip_len)
            indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
    
    return indices

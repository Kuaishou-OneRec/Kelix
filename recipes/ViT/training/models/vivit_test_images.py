import av
import numpy as np
import torch
import sys
import os
from PIL import Image

# 使用相对导入
from recipes.ViT.training.models.vivit.image_processing_vivit import VivitImageProcessor
from recipes.ViT.training.models.vivit.modeling_vivit import VivitModel
from huggingface_hub import hf_hub_download
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling
np.random.seed(0)


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


# file_path = hf_hub_download(
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )
# container = av.open(file_path)

# indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
# video = read_video_pyav(container=container, indices=indices)

image1 = Image.open('/llm_reco/luoxinchen/dataset/DenseFusion/data/images/DenseFusion-1M/000000/2396916002068.png').convert('RGB')
image2 = Image.open('/llm_reco/luoxinchen/dataset/DenseFusion/data/images/DenseFusion-1M/000000/2037287003965.png').convert('RGB')
frames1 = [image1]
frames2 = [image2]
videos = []
video = []
for frame in frames1:
    frame = np.array(frame)
    frame = resize(frame, size=(224, 224), resample=PILImageResampling.BILINEAR)
    video.append(frame)
videos.append(video)
video = []
for frame in frames2:
    frame = np.array(frame)
    frame = resize(frame, size=(224, 224), resample=PILImageResampling.BILINEAR)
    video.append(frame)
videos.append(video)


videos = [np.concatenate([frame[np.newaxis, ...] for frame in video],axis=0) for video in videos]
new_videos = []
for video in videos:
    indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=video.shape[0])
    video = read_image_pil(video,indices)
    video = list(video)
    new_videos.append(video)

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
inputs = image_processor(list(new_videos), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    pooler = outputs.pooler_output
    hidden_states = outputs.last_hidden_state
print('--------------------------------')
print(pooler.shape)
print('--------------------------------')
print(hidden_states.shape)
print('--------------------------------')
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])
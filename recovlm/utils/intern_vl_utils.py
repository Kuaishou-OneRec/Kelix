import os
import random
import re
from collections import Counter
from typing import Dict
import io
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import transformers
from decord import VideoReader
from PIL import Image
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
import transformers
from decord import VideoReader, cpu

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(is_train, input_size, pad2square=False, normalize_type='imagenet'):

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
    CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
    SIGLIP_MEAN = (0.5, 0.5, 0.5)
    SIGLIP_STD = (0.5, 0.5, 0.5)

    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    if is_train:  # use data augumentation
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        if pad2square is False:  # now we use this transform function by default
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])

    return transform

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    nframes,num_patches_list = [],[]
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        num_patches_list.append(len(img))
        nframes += img

    return nframes,num_patches_list


def process_vision_info_internvl(messages:list,
                                tokenizer: transformers.PreTrainedTokenizer,
                                visual_tokens_per_image:int,
                                min_dynamic_patch:int,
                                max_dynamic_patch:int,
                                use_thumbnail:bool,
                                image_size:int,
                                img_start_token:str,
                                img_context_token:str,
                                img_end_token:str,
                                normalize_type:str,
                                num_segments:int = 10
                                ):
    images = []
    iamge_tokens = ""
    new_conversations = []

    for conversation in messages:
      if conversation['role'] == "user":
        value = ""
        content = conversation["content"]
        for turn in content:
          if isinstance(turn, str):
            value += turn
          elif turn["type"] == "image":
            turn_images = dynamic_preprocess(turn["image"], min_num=min_dynamic_patch, max_num=max_dynamic_patch,
                                    image_size=image_size, use_thumbnail=use_thumbnail)
            images += [image for image in turn_images]
            num_image_tokens = visual_tokens_per_image * len(turn_images)
            value += f'{img_start_token}{img_context_token * num_image_tokens}{img_end_token}\n'

          elif turn['type'] == "video":
            nframes = []
            num_patches_list = []
            if isinstance(turn["video"], str) and "480p_60s_4fps" in turn["video"]:
                path = turn["video"]
                pid_str = osp.basename(osp.splitext(path)[0])
                if not osp.exists(path):
                    post = str(int(pid_str[-4:]))
                    path = path.replace("480p_60s_4fps_v2", "480p_60s_4fps_0215_0316/{}".format(post))
                nframes,num_patches_list = load_video(path,num_segments = num_segments)

            elif isinstance(turn["video"],list):
                for img in turn['video']:
                    imgs = dynamic_preprocess(img['image'], min_num=min_dynamic_patch, max_num=max_dynamic_patch,
                                    image_size=image_size, use_thumbnail=use_thumbnail)
                    num_patches_list.append(len(imgs))
                    nframes += imgs

            else:
                raise ValueError(f"process_vision_info_internvl failed,failed type {turn}")
            
            for i,num_image in enumerate(num_patches_list):
                #当前帧的token数
                num_image_tokens = visual_tokens_per_image * num_image
                value += f'Frame{i+1}: {img_start_token}{img_context_token * num_image_tokens}{img_end_token}\n'
                
            images += nframes
          elif turn["type"] == "text":
            value += turn["text"]
          else:
            raise ValueError(f"ERROR type {turn}")

        new_conversations.append({"role":"user","value":value})

      elif conversation["role"] == "assistant":
        value = ""
        content = conversation["content"]
        for turn in content:
          if isinstance(turn, str):
            value += turn
          elif isinstance(turn,dict):
            if turn["type"] == "text":
                value += turn["text"]
          else:
            raise ValueError(f"ERROR input type (assisant) {turn}")

        new_conversations.append({"role":"assistant","value":value})
      else:
        raise NotImplementedError
    image_flag = 1 if len(images) > 0 else 0
    #如果是纯文本增加一张图片做引导
    if image_flag==0:
      image = Image.new('RGB', (224, 224), (255, 255, 255))
      images = dynamic_preprocess(image, min_num=min_dynamic_patch, max_num=1,
                                        image_size=image_size, use_thumbnail=use_thumbnail)
    inputs = preprocess_internvl(new_conversations,tokenizer)
    transform = build_transform(is_train=True, input_size=image_size,normalize_type=normalize_type)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    inputs["pixel_values"] = pixel_values
    inputs["image_flags"] = torch.tensor([image_flag] * len(images), dtype=torch.long)

    return inputs

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'



from enum import IntEnum, auto
from typing import Any, Dict, List, Tuple, Union


def preprocess_internvl(conversations: list, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    #'你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
    system_prompt = 'You are a helpful assistant.'
    roles, batches = [], []
    
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    
    for conversation in conversations:
        if conversation['role'] == 'user':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('user')
        elif conversation['role'] == 'assistant':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('assistant')
        else:
            raise NotImplementedError
    
    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]
    
    tokenized_outputs = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        truncation=False,
    )
    
    input_ids = tokenized_outputs.input_ids
    
    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]
    
    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'user':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'assistant':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
    
    # 修复连接操作
    try:
        # 确保final_input_ids是一个可以连接的序列
        input_ids = torch.tensor(np.concatenate(final_input_ids))
        targets = torch.tensor(np.concatenate(final_targets))
    except TypeError:
        # 如果出现类型错误，可能是因为final_input_ids只包含一个数组
        if len(final_input_ids) == 1:
            input_ids = torch.tensor(final_input_ids[0])
            targets = torch.tensor(final_targets[0])
        else:
            # 尝试将final_input_ids和final_targets包装成list再连接
            input_ids = torch.tensor(np.concatenate([np.array(x) for x in final_input_ids]))
            targets = torch.tensor(np.concatenate([np.array(x) for x in final_targets]))
    
    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
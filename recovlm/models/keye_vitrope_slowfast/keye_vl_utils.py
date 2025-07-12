from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
import itertools
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import traceback
import io as py_io
import os.path as osp
import numpy as np
import copy


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 20480 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 32 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 65536 * 28 * 28
FPS = 2.0
FPS_MIN_FRAMES = 4

FAST_TOKEN_RATIO = 0.2

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
        height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    # if int(height < factor//4) + int(width < factor//4):
    #     raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor//4}")

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return max(h_bar, factor), max(w_bar, factor)


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR, open_fast_image = False) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")  ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        
        if open_fast_image:
            min_pixels = ele.get("video_min_pixels", VIDEO_MIN_PIXELS)
            max_pixels = ele.get("video_max_pixels", VIDEO_MAX_PIXELS)
        else:
            min_pixels = ele.get("min_pixels", MIN_PIXELS)
            max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    slow_image = image.resize((resized_width, resized_height))

    return slow_image



def smart_nframes(
        ele: dict,
        total_frames: int,
        video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = ele["nframes"]
    else:
        fps = ele.get("fps", FPS) # 应该是走的默认FPS，按照每秒抽两帧来算
        fps = min(fps, video_fps) # 注意，这里的video_fps是真实的后验FPS
        # print("cjx smart nfram debug VIDEO_TOTAL_PIXELS token num in llm side is {}".format(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS)//28//28))
        max_frames = int(ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS) / ele.get("video_min_pixels", VIDEO_MIN_PIXELS)) # 计算我们在fast设置下最多能吃多少帧，这个是用来兜底的
        fps_nframes = int(total_frames / video_fps * fps) # 换算为秒数，之后计算希望抽多少帧
        nframes = min(fps_nframes, max_frames)
    return nframes


def _read_video_torchvision(
        ele: dict,
) -> tuple[torch.Tensor, float]:
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    # process video url
    st = time.time()
    if isinstance(ele["video"], str):
        video_path = ele["video"]
        if version.parse(torchvision.__version__) < version.parse("0.19.0"):
            if "http://" in video_path or "https://" in video_path:
                warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
            if "file://" in video_path:
                video_path = video_path[7:]
        video, audio, info = io.read_video(
            video_path,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        total_frames, video_fps = video.size(0), info["video_fps"]
        logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    elif isinstance(ele["video"], bytes):
        video_reader = torchvision.io.VideoReader(ele["video"], "video")
        video_meta = video_reader.get_metadata()["video"]

        start_ptr = ele.get("video_start", 0.0)
        end_pts = ele.get("video_end", video_meta["duration"][-1])
        video = []
        for frame in itertools.takewhile(lambda x: x['pts'] <= end_pts, video_reader.seek(start_ptr)):
            video.append(frame['data'])
        video = torch.stack(video)
        total_frames, video_fps = video.size(0), video_meta["fps"][-1]
        logger.info(f"torchvision:  {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    nframes, fps_ratio = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    return video, fps_ratio


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
        ele: dict,
) -> tuple[torch.Tensor, float]:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    st = time.time()
    if isinstance(ele["video"], bytes):
        video_path = ""
        fp = py_io.BytesIO(ele["video"])
        vr = decord.VideoReader(fp)
    else:
        video_path = ele["video"]
        vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes, fps_ratio = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    return video, fps_ratio


def cal_sim(frame1, frame2, patch_size=28, pixel_threshold=5, patch_sim=0.99):
    assert frame1.dim() == 3 and frame2.dim() == 3, "输入必须是3D张量 [C, H, W]"
    
    channel, height, width = frame1.shape
    unchanged_threshold = patch_sim * channel * patch_size * patch_size
    
    from einops import rearrange
    
    diff = (frame1 - frame2).abs()
    unchanged_pixel = rearrange(diff < pixel_threshold, "c (h p1) (w p2) -> h w c p1 p2", p1=patch_size, p2=patch_size).long()

    patch_unchanged_count = unchanged_pixel.sum(-1).sum(-1).sum(-1)
    unchanged = (patch_unchanged_count.float() > unchanged_threshold)
    
    return unchanged.long().sum().item() / unchanged.numel()


def extract_key_frame(frames, patch_size=28, threshold=0.9):
    assert frames.dim() == 4, "输入必须是4D张量 [N, C, H, W]"
    
    key_frame_indices = [0]
    last_key_frame = frames[0]
    
    for i in range(1, frames.size(0)):
        current_frame = frames[i]
        
        global_sim = cal_sim(last_key_frame, current_frame)
        
        if global_sim < threshold:
            key_frame_indices.append(i)
            last_key_frame = current_frame  # 更新关键帧
            
    return key_frame_indices


def extract_slow_fast_frames(selected_frames, selected_frames_extract):
    print("selected_frames size {}, selected_frames_extract size {}".format(selected_frames.size(), selected_frames_extract.size()))
    slow_indices = extract_key_frame(selected_frames_extract)

    slow_mask = torch.zeros(size=(selected_frames.size(0), ), dtype=torch.bool)
    slow_mask[slow_indices] = True

    slow_frames = selected_frames[slow_mask]
    fast_frames = selected_frames[~slow_mask]

    slow_fast_order = torch.ones(size=(selected_frames.size(0), ), dtype=torch.long)
    slow_fast_order[slow_indices] = 0

    return slow_frames, fast_frames, slow_fast_order.tolist()


def _read_video_decord_slowfast(
        ele: dict,
) -> torch.Tensor:
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    st = time.time()
    if isinstance(ele["video"], bytes):
        video_path = ""
        fp = py_io.BytesIO(ele["video"])
        vr = decord.VideoReader(fp)
    else:
        video_path = ele["video"]
        vr = decord.VideoReader(video_path)
    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    total_frames_time_position = torch.FloatTensor([(1 / video_fps) * (i+1) for i in range(total_frames)])
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    total_nframes_number = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    
    selected_indices = torch.linspace(0, total_frames - 1, total_nframes_number).round().long()
    selected_frames = vr.get_batch(selected_indices.tolist()).asnumpy()
    selected_frames = torch.tensor(selected_frames).permute(0, 3, 1, 2)
    selected_time_position = total_frames_time_position[selected_indices]

    ##### extract key frames start ######
    # Step#1，对选中的图，假设都为slow，先resize到28*28的倍数
    _, _, height, width = selected_frames.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=ele.get("video_min_pixels", VIDEO_MIN_PIXELS),
        max_pixels=ele.get("video_max_pixels", VIDEO_MAX_PIXELS),
    )

    selected_frames_extract = nn.functional.interpolate(
        selected_frames,
        [resized_height, resized_width],
        mode="bicubic",
        antialias=True,
    ).float()
    # Step#2 对选中的图，筛选出其中关键帧部分，其余为slow
    slow_frames, fast_frames, slow_fast_order = extract_slow_fast_frames(selected_frames, selected_frames_extract)
    print("cjx vl debug for mp4, total_frames {}, total_nframes_number {}, slow frames {}, fast frames {}".format(total_frames, total_nframes_number, slow_frames.size(0), fast_frames.size(0)))
    ##### extract key frames start ######

    return slow_frames, fast_frames, selected_time_position.tolist(), slow_fast_order


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "slowfast_decord": _read_video_decord_slowfast,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    # return video_reader_backend
    # Hack
    return "slowfast_decord"



def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, slowfast: bool = True) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str) or isinstance(ele["video"], bytes):
        video_reader_backend = get_video_reader_backend()
        slow_frames, fast_frames, time_position, slow_fast_order = VIDEO_READER_BACKENDS[video_reader_backend](ele)

    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = []
        for video_element in ele["video"]:
            # preprocess images
            if isinstance(video_element, dict):
                images.append(fetch_image(video_element, size_factor=image_factor, open_fast_image = True))
            else:
                images.append(
                    fetch_image({"image": video_element, **process_info}, size_factor=image_factor, open_fast_image = True)
                )
        total_frames = len(images)
        
        tensor_images = [torch.from_numpy(np.array(pil_image.copy())).permute(2, 0, 1) for pil_image in images]
        tensor_images = torch.stack(tensor_images, dim=0)

        slow_frames, fast_frames, slow_fast_order = extract_slow_fast_frames(tensor_images, tensor_images.clone())
        time_position = None
    
    ### 计算slow fast的token量 begin ###
    slow_number = slow_frames.size(0)
    if fast_frames.size(0) == 0:
        fast_frames = None
    fast_number = fast_frames.size(0) if fast_frames is not None else 0

    left = ele.get("video_min_pixels", VIDEO_MIN_PIXELS) / 28 / 28
    right = ele.get("video_max_pixels", VIDEO_MAX_PIXELS) / 28 / 28
    while left < right:
        mid = int(left+right)//2
        if slow_number * mid * 28 * 28 + fast_number * max(int(0.2 * mid) * 28 * 28, ele.get("video_min_pixels", VIDEO_MIN_PIXELS)) > ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS):
            right = mid
        else:
            left = mid + 1
    slow_max_pixels = left * 28 * 28
    fast_max_pixels = max(int(0.2 * mid) * 28 * 28, ele.get("video_min_pixels", VIDEO_MIN_PIXELS))
    video_min_pixels = ele.get("video_min_pixels", VIDEO_MIN_PIXELS)
    ### 计算slow fast的token量 end ###

    nframes, _, height, width = slow_frames.shape

    #### slow part ######
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=video_min_pixels,
        max_pixels=slow_max_pixels,
    )

    fast_resized_height, fast_resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=video_min_pixels,
        max_pixels=fast_max_pixels,
    )

    if time_position is None: # image list
        slow_frames = []
        fast_frames = []
        for idx, value in enumerate(slow_fast_order):
            if value == 0:
                slow_frames.append(images[idx].resize((resized_width, resized_height)))
            else:
                fast_frames.append(images[idx].resize((fast_resized_width, fast_resized_height)))
        
        if len(fast_frames) == 0:
            fast_frames = None
        
        print("cjx vl debug for image list, slow frames {}, fast frames {}, slow token is {}, fast token is {}".format(len(slow_frames), len(fast_frames) if fast_frames is not None else 0, resized_height*resized_width//28//28, fast_resized_height*fast_resized_width//28//28))
        assert (len(slow_frames) if slow_frames is not None else 0) + (len(fast_frames) if fast_frames is not None else 0) == len(slow_fast_order)
        return slow_frames, fast_frames, slow_fast_order

    else: # mp4
        slow_frames = nn.functional.interpolate(
            slow_frames,
            [resized_height, resized_width],
            mode="bicubic",
            antialias=True,
        ).float()
        slow_frames = list(slow_frames.split(1, dim=0))
        #### fast part ######
        if fast_frames is not None:
            fast_frames = nn.functional.interpolate(
                fast_frames,
                [fast_resized_height, fast_resized_width],
                mode="bicubic",
                antialias=True,
            ).float()
            fast_frames = list(fast_frames.split(1, dim=0))
        
        print("cjx vl debug for mp4, slow frames {}, fast frames {}, slow token is {}, fast token is {}".format(len(slow_frames), len(fast_frames) if fast_frames is not None else 0, resized_height*resized_width//28//28, fast_resized_height*fast_resized_width//28//28))
        assert (len(slow_frames) if slow_frames is not None else 0) + (len(fast_frames) if fast_frames is not None else 0) == len(slow_fast_order)
        return slow_frames, fast_frames, time_position, slow_fast_order
    

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
        conversations: list[dict] | list[list[dict]] = None, vision_infos: list[dict] = None,
        image_factor: int = IMAGE_FACTOR
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    assert conversations is not None or vision_infos is not None

    if vision_infos is None:
        vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_factor))
        elif "video" in vision_info:
            if isinstance(vision_info["video"], str) and "480p_60s_4fps_v2" in vision_info["video"]:
                path = vision_info["video"]
                pid_str = osp.basename(osp.splitext(path)[0])
                if not osp.exists(path):
                    post = str(int(pid_str[-4:]))
                    path = path.replace("480p_60s_4fps_v2", "480p_60s_4fps_0215_0316/{}".format(post))
                vision_info["video"] = path
            video_inputs.append(fetch_video(vision_info, image_factor))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs



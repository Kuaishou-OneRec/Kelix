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
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import traceback
import io as py_io
import os.path as osp

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 32 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4


MAX_FRAMES = 40 # 总的frame数量
FPS_MAX_SLOW_FRAMES = 20 # 注意：这里的含义是Max Slow Frame，不是总的frames数量

FAST_IMAGE_FACTOR = 28
FAST_MIN_PIXELS = 1 * 28 * 28
FAST_MAX_PIXELS = 32 * 28 * 28
FAST_VIDEO_TOTAL_PIXELS = 8192 * 28 * 28

ONLY_SLOW = 0
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

    if open_fast_image:
        fast_min_pixels = ele.get("fast_min_pixels", FAST_MIN_PIXELS)
        fast_max_pixels = ele.get("fast_max_pixels", FAST_MAX_PIXELS)
        fast_resized_height, fast_resized_width = smart_resize(
            height,
            width,
            factor=FAST_IMAGE_FACTOR,
            min_pixels=fast_min_pixels,
            max_pixels=fast_max_pixels,
            )
        fast_image = image.resize((fast_resized_width, fast_resized_height))
        return slow_image, fast_image

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
        fps = ele.get("fps", FPS)
        min_frames = ele.get("min_frames", FPS_MIN_FRAMES)
        max_frames = ele.get("max_slow_frames", min(FPS_MAX_SLOW_FRAMES, total_frames))
        fps = min(fps, video_fps)
        nframes = total_frames / video_fps * fps
        nframes = int(min(max(nframes, min_frames), max_frames))
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
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    nframes, slow_fps_ratio = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps) # 这个地方已经是按照了
    slow_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    slow_frames = vr.get_batch(slow_idx).asnumpy()
    slow_frames = torch.tensor(slow_frames).permute(0, 3, 1, 2)  # Convert to TCHW format
    
    # TODO(caojiangxia): 这里可以稍微加一些随机性，见过更多样的slowfast比例
    if total_frames >= nframes * SLOWFAST_MAX_RATIO:
        fast_nframes = nframes * SLOWFAST_MAX_RATIO
        slowfast_rate = SLOWFAST_MAX_RATIO
    else:
        fast_nframes = total_frames
        slowfast_rate = total_frames / nframes

    fast_idx = torch.linspace(0, total_frames - 1, fast_nframes).round().long().tolist()
    filter_fast_idx = [x for x in fast_idx if x not in slow_idx]
    fast_frames = vr.get_batch(filter_fast_idx).asnumpy()
    fast_frames = torch.tensor(fast_frames).permute(0, 3, 1, 2)
    fast_fps_ratio = slow_fps_ratio * slowfast_rate

    mix_index = slow_idx + filter_fast_idx
    sort_mix_index = sorted(mix_index)
    slow_fast_order = [0 if index in slow_idx else 1 for index in sort_mix_index]
    return slow_frames, fast_frames, slow_fast_order, slow_fps_ratio, fast_fps_ratio



def _read_video_decord_slowfast_v2(
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

    slow_nframes_number = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    if ele.get("only_slow", ONLY_SLOW):
        print("cjx debug only slow True, max_slow_frames is {}".format(ele.get("max_slow_frames", FPS_MAX_SLOW_FRAMES)))
        fast_nframes_number = 0
    else:
        max_fast_frame_number = ele.get("max_frames", MAX_FRAMES) - slow_nframes_number
        fast_nframes_number = min(total_frames - slow_nframes_number, max_fast_frame_number)

    total_nframes_number = slow_nframes_number + fast_nframes_number
    selected_indices = torch.linspace(0, total_frames - 1, total_nframes_number).round().long()
    selected_time_position = total_frames_time_position[selected_indices]

    slow_indices = torch.linspace(0, total_nframes_number - 1, slow_nframes_number).round().long()
    slow_mask = torch.zeros(size=(total_nframes_number, ), dtype=torch.bool)
    slow_mask[slow_indices] = True

    selected_frames = vr.get_batch(selected_indices.tolist()).asnumpy()
    selected_frames = torch.tensor(selected_frames).permute(0, 3, 1, 2)
    slow_frames = selected_frames[slow_mask]
    fast_frames = selected_frames[~slow_mask] if fast_nframes_number > 0 else None

    slow_fast_order = torch.ones(size=(total_nframes_number, ), dtype=torch.long)
    slow_fast_order[slow_indices] = 0

    return slow_frames, fast_frames, selected_time_position.tolist(), slow_fast_order.tolist()
    





    


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
    "slowfast_decord": _read_video_decord_slowfast_v2,
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
        
        if image_factor is None:
            return None

        nframes, _, height, width = slow_frames.shape

        #### slow part ######
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels_1 = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_2 = ele.get("max_pixels", max_pixels_1)
        max_pixels = min(max_pixels_1, max_pixels_2)
        fast_min_pixels = ele.get("fast_min_pixels", FAST_MIN_PIXELS)
        fast_max_pixels = ele.get("fast_max_pixels", FAST_MAX_PIXELS)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )

            fast_resized_height, fast_resized_width = smart_resize(
                height,
                width,
                factor=FAST_IMAGE_FACTOR,
                min_pixels=fast_min_pixels,
                max_pixels=fast_max_pixels,
            )
        else:
            min_pixels = min(min_pixels, max_pixels)
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            fast_resized_height, fast_resized_width = smart_resize(
                height,
                width,
                factor=FAST_IMAGE_FACTOR,
                min_pixels=fast_min_pixels,
                max_pixels=fast_max_pixels,
            )

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

        assert (len(slow_frames) if slow_frames is not None else 0) + (len(fast_frames) if fast_frames is not None else 0) == len(slow_fast_order)
        # assert (slow_frames.size(0) if slow_frames is not None else 0) + (fast_frames.size(0) if fast_frames is not None else 0) == len(slow_fast_order)

        return slow_frames, fast_frames, time_position, slow_fast_order

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
        
        slow_nframes_number = ele.get("max_slow_frames", min(FPS_MAX_SLOW_FRAMES, total_frames))
        slow_idx = torch.linspace(0, total_frames - 1, slow_nframes_number).round().long().tolist()
        
        slow_frames = [images[idx][0] for idx in slow_idx]

        max_fast_frame_number = ele.get("max_frames", MAX_FRAMES) - slow_nframes_number
        fast_nframes_number = min(total_frames - slow_nframes_number, max_fast_frame_number)
        if  ele.get("only_slow", ONLY_SLOW):
            print("cjx debug only slow True, max_slow_frames is {}".format(ele.get("max_slow_frames", FPS_MAX_SLOW_FRAMES)))
            fast_nframes_number = 0
        if fast_nframes_number > 0:
            left_frame_list = [x for x in range(total_frames) if x not in slow_idx]

            left_fast_idx = torch.linspace(0, len(left_frame_list) - 1, fast_nframes_number).round().long().tolist()
            fast_idx = [left_frame_list[left_fast_idx[idx]] for idx in range(fast_nframes_number)]

            fast_frames = [images[idx][1] for idx in fast_idx]
            selected_index = slow_idx + fast_idx
            sort_selected_index = sorted(selected_index)
            slow_fast_order = [0 if index in slow_idx else 1 for index in sort_selected_index]
        else:
            fast_frames = None
            fast_time_position = []
            slow_fast_order = [0] * len(slow_frames)

        return slow_frames, fast_frames, slow_fast_order


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



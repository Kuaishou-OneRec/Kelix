from . import BaseHook
import torch
from einops import rearrange
import numpy as np
import json
from PIL import Image
from recipes.ViT.helpers.hook.utils import process_vision_info


class VisionEncoderDataProcessorHook(BaseHook):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resized_height = kwargs["vision"].resized_height
        self.resized_width = kwargs["vision"].resized_width
        self.multiple_of = kwargs["vision"].multiple_of
        self.min_pixels = kwargs["vision"].min_pixels
        self.max_pixels = kwargs["vision"].max_pixels
        self.video_min_pixels = kwargs["vision"].video_min_pixels
        self.video_max_pixels = kwargs["vision"].video_max_pixels
        self.video_total_pixels = kwargs["vision"].video_total_pixels
        self.nframes = kwargs["vision"].nframes
        self.fps = kwargs["vision"].fps
        self.fps_min_frames = kwargs["vision"].fps_min_frames
        self.fps_max_frames = kwargs["vision"].fps_max_frames

    def __call__(self, sample, row_info_str):
        images = None
        videos = None

        if "images" in sample:
            images = sample["images"]
            if isinstance(images, str):
                images = json.loads(images)

        if "videos" in sample:
            videos = sample["videos"]
            if isinstance(videos, str):
                videos = json.loads(videos)

        data_source = sample["source"]
        key = sample["uuid"]
        task = sample["task"]
        text = sample["text"]

        assert not (text[0] == '[' and text[-1] == ']')
    
        if isinstance(text, str):
            text = [text]

        processed_sample = {
            "source": data_source,
            "task": task,
            "texts": text,
            "uuid": key
        }

        vision_infos = list()

        if images is not None:
            if isinstance(images, list):
                pass
            elif isinstance(images, np.ndarray):
                images = images.tolist()
            else:
                raise NotImplementedError(f"Unsupported sample, images type is {type(images)}, images={images}")

            for image_block in images:
                if not isinstance(image_block, dict):
                    image_info = {
                        "image": image_block,
                    }
                else:
                    image_info = image_block
                if "resized_height" not in image_info and self.resized_height is not None:
                    image_info["resized_height"] = self.resized_height
                
                if "resized_width" not in image_info and self.resized_width is not None:
                    image_info["resized_width"] = self.resized_width

                vision_infos.append(image_info)

        if videos is not None:
            if isinstance(videos, list):
                pass
            elif isinstance(videos, np.ndarray):
                videos = videos.tolist()
            else:
                raise NotImplementedError(f"Unsupported sample, videos type is {type(videos)}, videos={videos}")

            for video_block in videos:

                if not isinstance(video_block, dict):
                    video_info = {
                        "video": video_block,
                    }
                else:
                    video_info = video_block
                
                if "nframes" not in video_info and self.nframes is not None:
                    video_info["nframes"] = self.nframes
                
                if "nframes" not in video_info and "fps" not in video_info and self.fps is not None:
                    video_info["fps"] = self.fps
                    video_info["min_frames"] = self.fps_min_frames
                    video_info["max_frames"] = self.fps_max_frames
                
                if "min_pixels" not in video_info and self.video_min_pixels is not None:
                    video_info["min_pixels"] = self.video_min_pixels
                
                if "max_pixels" not in video_info and self.video_max_pixels is not None:
                    video_info["max_pixels"] = self.video_max_pixels
                
                if "total_pixels" not in video_info and self.video_total_pixels is not None:
                    video_info["total_pixels"] = self.video_total_pixels

        images = process_vision_info(
            vision_infos, 
            multiple_of=self.multiple_of, 
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        processed_sample["images"] = images

        return processed_sample

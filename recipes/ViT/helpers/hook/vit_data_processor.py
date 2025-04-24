from . import BaseHook
import torch
from einops import rearrange
import numpy as np
from PIL import Image
from recipes.ViT.helpers.hook.utils import process_image_block, process_video_block


class VisionEncoderDataProcessorHook(BaseHook):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.force_size = 

    def __call__(self, sample, url):
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

        samples = {
            "__key__": key,
            "__url__": url,
        }

        sample_data = {
            "source": data_source,
            "task": task,
            "texts": text
        }

        if images is not None:
            if isinstance(images, list):
                pass
            elif isinstance(images, np.ndarray):
                images = images.tolist()
            else:
                raise NotImplementedError(f"Unsupported sample, images type is {type(images)}, images={images}")

            for image_idx, image_block in enumerate(images):
                image_obj = process_image_block(image_block)
                if image_obj is None:
                    return None
                images[image_idx] = image_obj

        if videos is not None:
            if isinstance(videos, list):
                pass
            elif isinstance(videos, np.ndarray):
                videos = videos.tolist()
            else:
                raise NotImplementedError(f"Unsupported sample, videos type is {type(videos)}, videos={videos}")

            for video_idx, video_block in enumerate(videos):
                video_obj = process_video_block(video_block)
                if video_obj is None:
                    return None
                videos[video_idx] = video_obj

        if images is not None and isinstance(images, list):
            sample_data["images"] = images
        elif videos is not None and isinstance(videos, list):
            sample_data["videos"] = videos
        elif images is not None and isinstance(images, np.ndarray):
            sample_data["images"] = images.tolist()
        elif videos is not None and isinstance(videos, np.ndarray):
            sample_data["videos"] = videos.tolist()
        else:
            raise NotImplementedError(
                f"Unsupported sample, images type is {type(images)}, images={images}, videos type is {type(videos)}, videos={videos}")
        samples["json"] = sample_data

        return samples

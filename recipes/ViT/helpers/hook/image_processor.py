from . import BaseHook
import torch
from einops import rearrange
import numpy as np
from PIL import Image


class ImageProcessorHook(BaseHook):

    def __init__(self, processor, **kwargs):
        super().__init__(processor, **kwargs)
        self.processor = processor
        packing_kwargs = kwargs.get("packing")
        self.add_cls_token = packing_kwargs.get('add_cls_token', False)
        self.drop_ratio = packing_kwargs.get('drop_ratio', 0.0)
        self.patch_size = packing_kwargs.get('patch_size')

    def calcul_image_tokens(self, image_or_image_list):
        """
        计算图片的token数
        """
        if isinstance(image_or_image_list, Image.Image):
            image = image_or_image_list
            width, height = image.size
            assert width % self.patch_size == 0 and height % self.patch_size == 0, image.size
            num_token = (width // self.patch_size) * (height // self.patch_size)
            return num_token

        num_token_list = list()
        for image_idx, image_obj in enumerate(image_or_image_list):
            assert isinstance(image_obj, Image.Image)
            num_token = self.calcul_image_tokens(image_obj)
            num_token_list.append(num_token)
        return num_token_list

    def __call__(self, sample):
        images = sample["json"]['images']
        patch_size = self.patch_size

        if isinstance(images, Image.Image):
            image_list = [images]
        elif isinstance(images, (list, tuple)):
            image_list = list(images)
        else:
            raise TypeError(f"Not supported image type '{images.__class__}'")

        total_token = 0
        image_indices = list()
        position_ids = list()
        height_position_ids = list()
        width_position_ids = list()
        pixel_values = list()

        for image_idx, image in enumerate(image_list):
            img_n_token = self.calcul_image_tokens(image)
            width, height = image.size
            img_n_token_after_drop = max(int(img_n_token * (1. - self.drop_ratio)), 1)

            total_token += img_n_token_after_drop

            image_indices.extend([image_idx for _ in range(img_n_token_after_drop)])

            positions = np.random.choice(img_n_token, img_n_token_after_drop, replace=False)
            positions = np.sort(positions)

            height_positions = positions // width
            width_positions = positions % width

            position_ids.extend(list(positions))
            height_position_ids.extend(list(height_positions))
            width_position_ids.extend(list(width_positions))

            processed_image = self.processor(images=image, return_tensors="pt", do_resize=False)
            pixels = processed_image["pixel_values"]
            assert pixels.dim() == 4 and pixels.shape[0] == 1
            pixels = pixels.squeeze(0)
            pixels = rearrange(pixels, "c (h p1) (w p2) -> (h w) c p1 p2", p1=patch_size, p2=patch_size)
            pixels = pixels[torch.LongTensor(positions)]
            pixel_values.append(pixels)

        if self.add_cls_token:
            padding_embedding = pixel_values[0].new_zeros(size=(1, 3, patch_size, patch_size))

            total_token += 1

            image_indices.append(max(image_indices) + 1)

            position_ids.append(-1)
            height_position_ids.append(-1)
            width_position_ids.append(-1)
            pixel_values.append(padding_embedding)

        image_indices = torch.LongTensor(image_indices)

        position_ids = torch.LongTensor(position_ids)
        height_position_ids = torch.LongTensor(height_position_ids)
        width_position_ids = torch.LongTensor(width_position_ids)
        pixel_values = torch.concat(pixel_values, dim=0)
        sample["json"].update(
            dict(
                image_indices=image_indices,
                position_ids=position_ids,
                height_position_ids=height_position_ids,
                width_position_ids=width_position_ids,
                pixel_values=pixel_values,
                images=image_list,
                seqlen=total_token
            )
        )
        return sample
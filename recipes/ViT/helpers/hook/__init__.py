from .base import BaseHook
from .image_processor import ImageProcessorHook
from .naive import NaiveHook
from .vision_encoder import VisionEncoderDataProcessorHook
from recipes.ViT.helpers.common import filter_function_arguments


def build_hook(**kwargs):

    hook_class = eval(kwargs.get("type"))
    init_kwargs = filter_function_arguments(hook_class.__init__, kwargs, new_obj=False, exclude_keys=["type"])

    hook = hook_class(**init_kwargs)
    
    return hook

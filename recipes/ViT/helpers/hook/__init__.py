from .base import BaseHook
from .image_processor import ImageProcessorHook
from .naive import NaiveHook
from recipes.ViT.helpers.common import filter_function_arguments


def build_hook(processor=None, **kwargs):

    hook_class = eval(kwargs.get("type"))
    init_kwargs = filter_function_arguments(hook_class.__init__, kwargs, new_obj=True, exclude_keys=["type"])

    hook = hook_class(processor=processor, **init_kwargs)
    
    return hook

from recipes.ViT.common import filter_function_arguments
from .base import BaseVerbose
from .vit import ViTVerbose


def build_verbose(config, ctx, **kwargs):
    verbose_class = eval(config.type)

    kwargs = filter_function_arguments(verbose_class.__init__, kwargs, new_obj=True)
    return verbose_class(config, ctx, **kwargs)

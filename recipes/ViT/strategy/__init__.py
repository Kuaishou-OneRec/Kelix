from recipes.ViT.common import filter_function_arguments
from .base import BaseStrategy
from .vit import ViTStrategy


def build_strategy(config, ctx, **kwargs):
    strategy_class = eval(config.type)

    kwargs = filter_function_arguments(strategy_class.__init__, kwargs, new_obj=True)
    return strategy_class(config, ctx, **kwargs)

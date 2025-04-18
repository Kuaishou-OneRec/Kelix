from .base import BaseMonitor
from .vit import ViTMonitor
from recipes.ViT.helpers.common import filter_function_arguments


def build_monitor(config, ctx, model, dataloader, **kwargs):
    monitor_class = eval(config.monitor.type)
    monitor = monitor_class(config, ctx, model, dataloader, **kwargs)
    return monitor

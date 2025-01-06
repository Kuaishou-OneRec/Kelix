from .the_cauldron_converter import TheCaulDronConverter
from .converter import EmptyConverter
from .kwai_video import KwaiVideoCaptionConverter


def create_converter(cfg):
    return eval(cfg.class_name)(**cfg.kwargs)
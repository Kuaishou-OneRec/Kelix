from .the_cauldron_converter import TheCaulDronConverter
from .converter import EmptyConverter
from .kwai_video import KwaiVideoCaptionConverter
from .dense_fusion_converter import DenseFusionConverter


def create_converter(cfg):
    return eval(cfg.class_name)(**cfg.kwargs)
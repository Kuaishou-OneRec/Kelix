from .the_cauldron_converter import TheCaulDronConverter
from .converter import EmptyConverter
from .kwai_video import KwaiVideoCaptionConverter
from .dense_fusion_converter import DenseFusionConverter
from .llava_cc3m_converter import LlavaCC3MPretrainConverter


def create_converter(cfg):
    return eval(cfg.class_name)(**cfg.kwargs)
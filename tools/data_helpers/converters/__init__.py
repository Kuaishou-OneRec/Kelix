from .converter import (
    ConverterBase,
    EmptyConverter
)
from .the_cauldron_converter import TheCaulDronConverter
from .kwai_video import KwaiVideoCaptionConverter
from .dense_fusion_converter import DenseFusionConverter
from .llava_cc3m_converter import LlavaCC3MPretrainConverter
from .doc_matrix_converter import DocmatrixConverter


def create_converter(cfg) -> ConverterBase:
    return eval(cfg.class_name)(**cfg.kwargs)
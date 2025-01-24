from .filter import FilterBase
from .alpha_numeric_filter import AlphaNumericFilter
from .image_size_filter import ImageSizeFilter
from .score_filter import ScoreFilter
from .text_length_filter import TextLengthFilter
from .special_characters_filter import SpecialCharactersFilter

def create_filter(cfg) -> FilterBase:
    return eval(cfg.class_name)(**cfg.kwargs)
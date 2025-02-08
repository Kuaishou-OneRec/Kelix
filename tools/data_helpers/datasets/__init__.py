from .the_cauldron_dataset import TheCaulDronDataset
from .dataset import (
    DistDataset,
    ParquetDataset,
    JsonlDataset,
    JsonDataset,
    WebDataset,
    TgzImageDataset,
    VlmTextJsonl,
    PubTabNetDataset
)
from .fintabnet import FinTabNetDataset
    
def create_dataset(cfg) -> DistDataset:
    return eval(cfg.class_name)(**cfg.kwargs)
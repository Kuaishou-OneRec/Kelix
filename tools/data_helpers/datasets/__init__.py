from .the_cauldron_dataset import TheCaulDronDataset
from .dataset import (
    ParquetDataset,
    JsonlDataset
)
    
def create_dataset(cfg):
    return eval(cfg.class_name)(**cfg.kwargs)
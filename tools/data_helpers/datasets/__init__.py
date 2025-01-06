from .the_cauldron_dataset import TheCaulDronDataset
from .dataset import ParquetDataset
    
def create_dataset(cfg):
    return eval(cfg.class_name)(**cfg.kwargs)
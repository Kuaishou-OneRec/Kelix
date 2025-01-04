from .the_cauldron_dataset import TheCaulDronDataset
    
def create_dataset(cfg):
    return eval(cfg.class_name)(**cfg.kwargs)
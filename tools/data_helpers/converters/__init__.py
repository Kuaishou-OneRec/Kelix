from .the_cauldron_converter import TheCaulDronConverter

def create_converter(cfg):
    return eval(cfg.class_name)(**cfg.kwargs)
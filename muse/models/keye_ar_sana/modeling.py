from muse.models.base import Model
from muse.config.model_config import KeyeARSANAConfig



class KeyeARSANAModel(Model):
    def __init__(self, config: KeyeARSANAConfig):
        super().__init__(config)
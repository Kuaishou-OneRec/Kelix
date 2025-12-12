from typing import Dict, Callable, List, Optional, Tuple
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging
from einops import rearrange

from muse.models.base import Model
from muse.config import  KeyeVisionConfig
from muse.config.model_config import ModelConfig, KeyeTokenizerConfig
from muse.models.keye_vit.modeling import KeyeVisionTransformer

# Import will be done when muse.models is imported, avoiding circular import
# The actual registration happens in __init__.py after import

logger = logging.getLogger(__name__)



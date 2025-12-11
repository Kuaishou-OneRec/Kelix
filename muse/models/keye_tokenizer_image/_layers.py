# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Literal

import torch
from torch import nn
from muse.layers.attention_utils import get_attention_function
from muse.layers.feed_forward import FeedForward
from muse.layers.kv_cache import KVCache

from muse.training.parallel import get_context_parallel_world_size, \
    get_context_parallel_group, SeqAllToAll4D

logger = logging.getLogger(__name__)



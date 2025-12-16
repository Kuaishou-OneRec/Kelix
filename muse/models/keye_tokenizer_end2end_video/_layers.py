# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Literal, Tuple, List

import torch
from torch import nn
from muse.layers.attention_utils import get_attention_function
from muse.layers.feed_forward import FeedForward
from muse.layers.kv_cache import KVCache
from typing import Tuple, List


class Projector(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 merge_kernel_size: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size

        self.hidden_size = (
            in_channels * self.merge_kernel_size[0] * self.merge_kernel_size[1]
        )

        self.pre_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            self.hidden_size, out_channels, bias=True
        )

    def forward(self,
                image_features: torch.Tensor,
                image_grid_thw: List[Tuple[int, int, int]]) -> torch.Tensor:
        m1, m2 = self.merge_kernel_size

        if isinstance(image_features, (list, tuple)):
            processed_features = list()
            for image_feature, image_grid in zip(image_features, image_grid_thw):
                t, h, w = image_grid
                from einops import rearrange
                image_feature = rearrange(image_feature, "(t h p1 w p2) d -> (t h w) (p1 p2 d)", t=t, h=h // m1, p1=m1, w=w // m2, p2=m2)
                image_feature = self.pre_norm(image_feature)
                hidden_states = self.linear_1(image_feature)
                hidden_states = self.act(hidden_states)
                hidden_states = self.linear_2(hidden_states)
                processed_features.append(hidden_states)
            processed_features = torch.concat(processed_features, dim=0)

            return processed_features

        # Fallback for single tensor input (should not reach here normally)
        assert image_features.dim() == 2, f"Expected 2D tensor, got {image_features.shape}"
        dim = image_features.shape[-1]
        hidden_states = self.pre_norm(image_features.view(-1, self.hidden_size))
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


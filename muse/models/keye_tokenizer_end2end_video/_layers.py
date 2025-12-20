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

import numpy as np
from einops import rearrange

class Projector(nn.Module):
    """
    视觉特征降采样/投影模块，移植自 origin，实现时序/空间合并。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_merge_position: str = "before",
        temporal_merge_mode: str = "avg",
        merge_kernel_size: Tuple[int, int] = (2, 2),
    ):
        super().__init__()
        self.merge_kernel_size = merge_kernel_size
        self.temporal_merge_position = temporal_merge_position
        self.temporal_merge_mode = temporal_merge_mode

        if self.temporal_merge_position not in ["before", "after"]:
            raise ValueError(f"Unsupported temporal_merge_position={self.temporal_merge_position}")
        if self.temporal_merge_position == "before" and self.temporal_merge_mode not in ["avg", "delta"]:
            raise ValueError("temporal_merge_mode must be 'avg' or 'delta' when temporal_merge_position='before'")
        if self.temporal_merge_position == "after" and self.temporal_merge_mode not in ["avg", "delta"]:
            raise ValueError("temporal_merge_mode must be 'avg' or 'delta' when temporal_merge_position='after'")

        self.hidden_size = in_channels * self.merge_kernel_size[0] * self.merge_kernel_size[1]
        self.temporal_delta_norm_before = None
        self.temporal_delta_rnn_before = None
        self.temporal_delta_norm_after = None
        self.temporal_delta_rnn_after = None
        if self.temporal_merge_position == "before" and self.temporal_merge_mode == "delta":
            self.temporal_delta_norm_before = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
            self.temporal_delta_rnn_before = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        if self.temporal_merge_position == "after" and self.temporal_merge_mode == "delta":
            self.temporal_delta_norm_after = torch.nn.LayerNorm(out_channels, eps=1e-05)
            self.temporal_delta_rnn_after = nn.GRU(
                input_size=out_channels,
                hidden_size=out_channels,
                num_layers=1,
                batch_first=True,
            )

        self.pre_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(self.hidden_size, out_channels, bias=True)

    def split(self, last_hidden_state: torch.Tensor, grid_thw: torch.Tensor):
        sample_hidden_state = list()
        lengths = np.prod(grid_thw.cpu().numpy(), axis=1).tolist()
        assert sum(lengths) == last_hidden_state.shape[1]
        start = 0
        for length in lengths:
            end = start + length
            tensor = last_hidden_state[:, start:end, :].squeeze(0)
            sample_hidden_state.append(tensor)
            start = end
        return sample_hidden_state

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.pre_norm(tokens)
        tokens = self.linear_1(tokens)
        tokens = self.act(tokens)
        tokens = self.linear_2(tokens)
        return tokens

    def _temporal_reduce_before(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_merge_mode == "avg":
            return x.mean(dim=0)
        if self.temporal_merge_mode == "delta":
            x = x.permute(1, 0, 2)  # (n, t, hidden)
            if x.size(1) == 1:
                x = x.repeat(1, 2, 1)
            x = self.temporal_delta_norm_before(x)
            _, h_n = self.temporal_delta_rnn_before(x)
            return h_n[-1]
        raise ValueError(f"Unsupported temporal_merge_mode {self.temporal_merge_mode} for before merge.")

    def _temporal_reduce_after(self, x: torch.Tensor, spatial_h: int, spatial_w: int) -> torch.Tensor:
        if self.temporal_merge_mode == "avg":
            return x.mean(dim=0)
        if self.temporal_merge_mode == "delta":
            x = x.permute(1, 0, 2)  # (n, t, hidden)
            if x.size(1) == 1:
                x = x.repeat(1, 2, 1)
            x = self.temporal_delta_norm_after(x)
            _, h_n = self.temporal_delta_rnn_after(x)
            return h_n[-1]

    def _process_sequence(self, image_feature: torch.Tensor, image_grid: Tuple[int, int, int]) -> torch.Tensor:
        t, h, w = [int(x) for x in image_grid]
        if t == 0:
            return image_feature.new_zeros((0, self.linear_2.out_features))
        m1, m2 = self.merge_kernel_size
        spatial_h = max(1, h // m1)
        spatial_w = max(1, w // m2)
        x = rearrange(
            image_feature,
            "(t h p1 w p2) d -> t (h w) (p1 p2 d)",
            t=t,
            h=spatial_h,
            p1=m1,
            w=spatial_w,
            p2=m2,
        )
        if self.temporal_merge_position == "before":
            reduced = self._temporal_reduce_before(x)
            reduced = reduced.reshape(-1, self.hidden_size)
            return self._project_tokens(reduced)
        projected = self._project_tokens(x.reshape(-1, self.hidden_size))
        projected = projected.view(t, spatial_h * spatial_w, -1)
        reduced = self._temporal_reduce_after(projected, spatial_h, spatial_w)
        return reduced.reshape(-1, projected.shape[-1])

    def forward(self, image_features: torch.Tensor, image_grid_thw: List[Tuple[int, int, int]]) -> torch.Tensor:
        image_grid_thw_tensor = torch.tensor(image_grid_thw, device=image_features.device)
        image_features = self.split(image_features, image_grid_thw_tensor)
        if isinstance(image_features, (list, tuple)):
            return torch.cat(
                [
                    self._process_sequence(image_feature, image_grid)
                    for image_feature, image_grid in zip(image_features, image_grid_thw)
                ],
                dim=0,
            )
        outputs = []
        start = 0
        m1, m2 = self.merge_kernel_size
        for grid in image_grid_thw:
            t, h, w = [int(x) for x in grid]
            spatial_h = max(1, h // m1)
            spatial_w = max(1, w // m2)
            num_tokens = t * spatial_h * spatial_w
            sample = image_features[start : start + num_tokens]
            start += num_tokens
            outputs.append(self._process_sequence(sample, (t, h, w)))
        return torch.cat(outputs, dim=0)
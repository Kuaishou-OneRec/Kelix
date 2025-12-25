# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
# Modified for muse framework
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Diffusion Loss Functions.

This module implements loss functions for diffusion model training,
following the exact logic from the Sana codebase.

Reference:
- Sana/diffusion/model/gaussian_diffusion.py Lines 745-882
- Sana/diffusion/model/respace.py Lines 113-153
"""

from typing import Optional, Dict, Any, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute density for sampling timesteps during training.
    
    Reference: Sana/diffusion/model/respace.py Lines 113-153
    
    Args:
        weighting_scheme: Sampling scheme ('logit_normal', 'mode', 'uniform')
        batch_size: Number of samples
        logit_mean: Mean for logit-normal distribution
        logit_std: Std for logit-normal distribution
        mode_scale: Scale for mode weighting
    
    Returns:
        Tensor of shape [batch_size] with values in [0, 1]
    """
    if weighting_scheme == "logit_normal":
        # SD3-style logit-normal sampling
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        # Uniform sampling
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


class FlowMatchingScheduler:
    """Flow Matching scheduler for diffusion training.
    
    This implements the flow matching formulation where:
    - x_t = (1-t) * x_0 + t * noise  (linear interpolation)
    - velocity = noise - x_0
    - model predicts velocity
    
    Reference: Sana/diffusion/model/gaussian_diffusion.py
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        flow_shift: float = 1.0,
    ):
        """Initialize scheduler.
        
        Args:
            num_timesteps: Number of training timesteps
            flow_shift: Flow shift parameter for timestep mapping
        """
        self.num_timesteps = num_timesteps
        self.flow_shift = flow_shift
        
        # Build sigma schedule for flow matching
        # Following Sana's linear_flow schedule:
        # betas go from 1.0 to 0.001, sigmas = 1 - betas
        # So sigmas go from 0 to 0.999 (clean to noise)
        # t=0 -> sigma=0 (clean image), t=999 -> sigma≈1 (pure noise)
        betas = np.linspace(1.0, 0.001, num_timesteps)
        sigmas = 1.0 - betas  # sigmas: 0 -> 0.999
        
        # Apply flow shift to sigmas (for q_sample interpolation)
        if flow_shift != 1.0:
            sigmas = flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)
        
        self.sigmas = torch.from_numpy(sigmas).float()
        self.alphas = 1.0 - self.sigmas
        
        # Build timestep_map for model input (following Sana's SpacedDiffusion._wrap_model)
        # The model expects timestep values mapped by: timestep_map[t] = sigmas[t] * num_timesteps
        # This is crucial for correct timestep embedding behavior!
        # Reference: Sana/diffusion/model/respace.py Lines 444-445, 497-498
        self.timestep_map = torch.from_numpy(sigmas * num_timesteps).float()
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_t given x_0 using flow matching interpolation.
        
        Args:
            x_start: Clean data [B, C, H, W]
            t: Timestep indices [B,]
            noise: Optional noise tensor
        
        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get alpha and sigma for each timestep
        # Cast to input dtype to avoid dtype mismatch with FSDP mixed precision
        alphas = self.alphas.to(device=x_start.device, dtype=x_start.dtype)[t]
        sigmas = self.sigmas.to(device=x_start.device, dtype=x_start.dtype)[t]
        
        # Reshape for broadcasting
        while alphas.dim() < x_start.dim():
            alphas = alphas.unsqueeze(-1)
            sigmas = sigmas.unsqueeze(-1)
        
        # Linear interpolation: x_t = alpha * x_0 + sigma * noise
        x_t = alphas * x_start + sigmas * noise
        
        return x_t
    
    def get_velocity_target(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Get velocity target for flow matching.
        
        Args:
            x_start: Clean data
            noise: Noise tensor
        
        Returns:
            Velocity target: noise - x_start
        """
        return noise - x_start


class FlowMatchingLoss(nn.Module):
    """Flow Matching loss for diffusion training.
    
    This computes the MSE loss between predicted and target velocity
    following the flow matching formulation.
    
    Reference: Sana/diffusion/model/gaussian_diffusion.py Lines 745-882
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        flow_shift: float = 3.0,
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        pred_sigma: bool = False,
    ):
        """Initialize loss.
        
        Args:
            num_timesteps: Number of training timesteps
            flow_shift: Flow shift parameter
            weighting_scheme: Timestep sampling scheme
            logit_mean: Mean for logit-normal sampling
            logit_std: Std for logit-normal sampling
            pred_sigma: Whether model predicts sigma (variance)
        """
        super().__init__()
        self.scheduler = FlowMatchingScheduler(num_timesteps, flow_shift)
        self.num_timesteps = num_timesteps
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.pred_sigma = pred_sigma
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps for training.
        
        Args:
            batch_size: Number of samples
            device: Target device
        
        Returns:
            Timestep indices [batch_size,]
        """
        # Sample timesteps
        # Note: Each rank should have different timesteps. This is ensured by setting
        # rank-specific random seed at training start: seed = base_seed + rank
        u = compute_density_for_timestep_sampling(
            self.weighting_scheme,
            batch_size,
            self.logit_mean,
            self.logit_std,
        )
        timesteps = (u * self.num_timesteps).long().to(device)
        # Clamp to valid range
        timesteps = timesteps.clamp(0, self.num_timesteps - 1)
        return timesteps
    
    def forward(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute flow matching loss.
        
        Args:
            model: Diffusion model
            x_start: Clean data [B, C, H, W]
            y: Conditioning (text embeddings)
            mask: Attention mask
            timesteps: Optional pre-sampled timesteps
            noise: Optional pre-sampled noise
            model_kwargs: Additional model kwargs
        
        Returns:
            Dict with 'loss' and other terms
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timesteps if not provided
        # Use rank-specific generator for timestep sampling to ensure diversity across ranks
        if timesteps is None:
            timesteps = self.sample_timesteps(batch_size, device)
        
        # Sample noise
        # Note: Each rank should have different noise. This is ensured by setting
        # rank-specific random seed at training start: seed = base_seed + rank
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get noisy input
        x_t = self.scheduler.q_sample(x_start, timesteps, noise)
        
        # Get velocity target
        target = self.scheduler.get_velocity_target(x_start, noise)
        
        # Map timesteps for model input (following Sana's _WrappedModel)
        # The model expects mapped timesteps, not raw indices!
        # Reference: Sana/diffusion/model/respace.py Lines 497-498
        model_timesteps = self.scheduler.timestep_map.to(device=device, dtype=x_start.dtype)[timesteps]
        
        # Model prediction (use mapped timesteps!)
        model_output = model(x_t, model_timesteps, y, mask=mask, **model_kwargs)

        # Handle sigma prediction
        if self.pred_sigma:
            model_output, model_var = model_output.chunk(2, dim=1)
        
        # Compute MSE loss
        loss = (target - model_output) ** 2
        loss = mean_flat(loss)
        
        terms = {
            "loss": loss.mean(),
            "mse": loss.mean(),
            "noise": noise,
            "x_t": x_t,
            "model_output": model_output,
        }
        
        return terms


class DiffusionTrainer:
    """Helper class for diffusion training.
    
    This encapsulates the training logic including:
    - Timestep sampling
    - Loss computation
    - Model forward pass
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: FlowMatchingLoss,
    ):
        """Initialize trainer.
        
        Args:
            model: Diffusion model
            loss_fn: Loss function
        """
        self.model = model
        self.loss_fn = loss_fn
    
    def training_step(
        self,
        x_start: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Single training step.
        
        Args:
            x_start: Clean latents
            y: Text embeddings
            mask: Attention mask
        
        Returns:
            Loss dict
        """
        return self.loss_fn(self.model, x_start, y, mask, **kwargs)





# import time

# class FlowMatchingLoss(nn.Module):
#     """Flow Matching loss for diffusion training.
    
#     This computes the MSE loss between predicted and target velocity
#     following the flow matching formulation.
    
#     Reference: Sana/diffusion/model/gaussian_diffusion.py Lines 745-882
#     """
    
#     def __init__(
#         self,
#         num_timesteps: int = 1000,
#         flow_shift: float = 3.0,
#         weighting_scheme: str = "logit_normal",
#         logit_mean: float = 0.0,
#         logit_std: float = 1.0,
#         pred_sigma: bool = False,
#     ):
#         """Initialize loss.
        
#         Args:
#             num_timesteps: Number of training timesteps
#             flow_shift: Flow shift parameter
#             weighting_scheme: Timestep sampling scheme
#             logit_mean: Mean for logit-normal sampling
#             logit_std: Std for logit-normal sampling
#             pred_sigma: Whether model predicts sigma (variance)
#         """
#         super().__init__()
#         self.scheduler = FlowMatchingScheduler(num_timesteps, flow_shift)
#         self.num_timesteps = num_timesteps
#         self.weighting_scheme = weighting_scheme
#         self.logit_mean = logit_mean
#         self.logit_std = logit_std
#         self.pred_sigma = pred_sigma
        
#         # ========== 新增：计时相关初始化 ==========
#         self.forward_count = 0  # forward调用计数器
#         # 初始化各步骤耗时统计（累计值）
#         self.time_stats = {
#             "init_model_kwargs": 0.0,
#             "sample_timesteps": 0.0,
#             "sample_noise": 0.0,
#             "q_sample_x_t": 0.0,
#             "get_velocity_target": 0.0,
#             "map_timesteps": 0.0,
#             "model_forward": 0.0,
#             "handle_sigma": 0.0,
#             "compute_loss": 0.0,
#             "build_terms": 0.0,
#             "total_forward": 0.0,
#         }
    
#     # ========== 新增：打印耗时统计 ==========
#     def _print_time_stats(self):
#         """打印最近N次forward的耗时统计（平均）"""
#         print(f"\n===== FlowMatchingLoss 耗时统计 (累计{self.forward_count}次，最近10次平均) =====")
#         # 计算平均耗时（除以10次）
#         avg_stats = {k: v / 10 for k, v in self.time_stats.items()}
        
#         # 按耗时从大到小排序打印
#         sorted_items = sorted(avg_stats.items(), key=lambda x: x[1], reverse=True)
#         for step_name, avg_time in sorted_items:
#             # 格式化输出（毫秒级显示，更直观）
#             print(f"  {step_name:<20}: {avg_time * 1000:.4f} ms")
    
#     # ========== 新增：重置耗时统计 ==========
#     def _reset_time_stats(self):
#         """重置耗时统计（为下10次累计做准备）"""
#         for k in self.time_stats.keys():
#             self.time_stats[k] = 0.0
    
#     def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
#         """Sample timesteps for training.
        
#         Args:
#             batch_size: Number of samples
#             device: Target device
        
#         Returns:
#             Timestep indices [batch_size,]
#         """
#         # 占位：原代码中compute_density_for_timestep_sampling需替换为实际实现
#         def compute_density_for_timestep_sampling(weighting_scheme, batch_size, logit_mean, logit_std):
#             return torch.rand(batch_size)
        
#         # Sample timesteps
#         # Note: Each rank should have different timesteps. This is ensured by setting
#         # rank-specific random seed at training start: seed = base_seed + rank
#         u = compute_density_for_timestep_sampling(
#             self.weighting_scheme,
#             batch_size,
#             self.logit_mean,
#             self.logit_std,
#         )
#         timesteps = (u * self.num_timesteps).long().to(device)
#         # Clamp to valid range
#         timesteps = timesteps.clamp(0, self.num_timesteps - 1)
#         return timesteps
    
#     def forward(
#         self,
#         model: nn.Module,
#         x_start: torch.Tensor,
#         y: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         timesteps: Optional[torch.Tensor] = None,
#         noise: Optional[torch.Tensor] = None,
#         model_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> Dict[str, torch.Tensor]:
#         """Compute flow matching loss.
        
#         Args:
#             model: Diffusion model
#             x_start: Clean data [B, C, H, W]
#             y: Conditioning (text embeddings)
#             mask: Attention mask
#             timesteps: Optional pre-sampled timesteps
#             noise: Optional pre-sampled noise
#             model_kwargs: Additional model kwargs
        
#         Returns:
#             Dict with 'loss' and other terms
#         """
#         # ========== 总计时开始 ==========
#         torch.cuda.synchronize() if x_start.is_cuda else None  # GPU同步，确保计时准确
#         total_start = time.time()
        
#         # ========== 步骤1：初始化model_kwargs ==========
#         step_start = time.time()
#         if model_kwargs is None:
#             model_kwargs = {}
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["init_model_kwargs"] += time.time() - step_start
        
#         batch_size = x_start.shape[0]
#         device = x_start.device
        
#         # ========== 步骤2：采样timesteps（如果需要） ==========
#         step_start = time.time()
#         if timesteps is None:
#             timesteps = self.sample_timesteps(batch_size, device)
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["sample_timesteps"] += time.time() - step_start
        
#         # ========== 步骤3：采样noise（如果需要） ==========
#         step_start = time.time()
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["sample_noise"] += time.time() - step_start

#         # ========== 步骤4：计算x_t（q_sample） ==========
#         step_start = time.time()
#         x_t = self.scheduler.q_sample(x_start, timesteps, noise)
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["q_sample_x_t"] += time.time() - step_start
        
#         # ========== 步骤5：获取velocity target ==========
#         step_start = time.time()
#         target = self.scheduler.get_velocity_target(x_start, noise)
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["get_velocity_target"] += time.time() - step_start
        
#         # ========== 步骤6：映射timesteps ==========
#         step_start = time.time()
#         model_timesteps = self.scheduler.timestep_map.to(device=device, dtype=x_start.dtype)[timesteps]
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["map_timesteps"] += time.time() - step_start
        
#         # ========== 步骤7：模型前向预测 ==========
#         step_start = time.time()
#         model_output = model(x_t, model_timesteps, y, mask=mask,** model_kwargs)
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["model_forward"] += time.time() - step_start

#         # ========== 步骤8：处理sigma预测 ==========
#         step_start = time.time()
#         if self.pred_sigma:
#             model_output, model_var = model_output.chunk(2, dim=1)
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["handle_sigma"] += time.time() - step_start
        
#         # ========== 步骤9：计算MSE loss ==========
#         step_start = time.time()
#         loss = (target - model_output) **2
#         loss = mean_flat(loss)
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["compute_loss"] += time.time() - step_start
        
#         # ========== 步骤10：构建返回字典 ==========
#         step_start = time.time()
#         terms = {
#             "loss": loss.mean(),
#             "mse": loss.mean(),
#             "noise": noise,
#             "x_t": x_t,
#             "model_output": model_output,
#         }
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["build_terms"] += time.time() - step_start
        
#         # ========== 总计时结束 ==========
#         torch.cuda.synchronize() if x_start.is_cuda else None
#         self.time_stats["total_forward"] += time.time() - total_start
        
#         # ========== 计数器更新 + 每10次打印统计 ==========
#         self.forward_count += 1
#         if self.forward_count % 10 == 0:
#             self._print_time_stats()
#             self._reset_time_stats()  # 重置统计，为下10次累计
        
#         return terms
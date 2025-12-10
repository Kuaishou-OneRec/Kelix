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
Flow-DPM-Solver for accelerated sampling of Flow Matching models.

This module implements a simplified version of DPM-Solver specifically optimized
for Flow Matching models like Sana. It provides faster sampling compared to
Euler methods by using higher-order ODE solvers.

Reference: https://github.com/NVlabs/Sana
Based on: Sana/diffusion/model/dpm_solver.py
"""

import os
from typing import Callable, Dict, List, Optional, Any

import torch
from tqdm import tqdm


def expand_dims(v: torch.Tensor, dims: int) -> torch.Tensor:
    """Expand the tensor `v` to the dim `dims`.
    
    Args:
        v: A pytorch tensor with shape [N].
        dims: Target number of dimensions.
    
    Returns:
        A pytorch tensor with shape [N, 1, 1, ..., 1] with `dims` dimensions.
    """
    return v[(...,) + (None,) * (dims - 1)]


class NoiseScheduleFlow:
    """Noise schedule for Flow Matching models.
    
    Flow Matching uses a simple linear interpolation:
    - alpha_t = 1 - t (signal coefficient)
    - sigma_t = t (noise coefficient)
    
    This gives x_t = (1-t) * x_0 + t * noise
    """
    
    def __init__(self, schedule: str = "discrete_flow"):
        """Initialize the noise schedule.
        
        Args:
            schedule: Schedule type, should be "discrete_flow"
        """
        self.T = 1
        self.t0 = 0.001
        self.schedule = schedule
        self.total_N = 1000

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Compute log(alpha_t) of a given continuous-time label t in [0, T]."""
        return torch.log(self.marginal_alpha(t))

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha_t of a given continuous-time label t in [0, T]."""
        return 1 - t

    @staticmethod
    def marginal_std(t: torch.Tensor) -> torch.Tensor:
        """Compute sigma_t of a given continuous-time label t in [0, T]."""
        return t

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Compute lambda_t = log(alpha_t) - log(sigma_t) of a given t in [0, T]."""
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = torch.log(self.marginal_std(t))
        return log_mean_coeff - log_std

    @staticmethod
    def inverse_lambda(lamb: torch.Tensor) -> torch.Tensor:
        """Compute the continuous-time label t given half-logSNR lambda_t."""
        return torch.exp(-lamb)


def model_wrapper(
    model: Callable,
    noise_schedule: NoiseScheduleFlow,
    model_type: str = "flow",
    model_kwargs: Optional[Dict] = None,
    guidance_type: str = "classifier-free",
    condition: Optional[torch.Tensor] = None,
    unconditional_condition: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
) -> Callable:
    """Create a wrapper function for the noise prediction model.
    
    DPM-Solver needs a continuous-time model function. This wrapper converts
    the discrete-time model to a continuous-time noise prediction function
    and handles classifier-free guidance.
    
    Args:
        model: The diffusion model function.
        noise_schedule: The noise schedule object.
        model_type: Type of model output ("flow" for velocity prediction).
        model_kwargs: Additional kwargs passed to the model.
        guidance_type: Type of guidance ("uncond" or "classifier-free").
        condition: Conditional embeddings.
        unconditional_condition: Unconditional embeddings for CFG.
        guidance_scale: CFG scale.
    
    Returns:
        A wrapped model function for DPM-Solver.
    """
    if model_kwargs is None:
        model_kwargs = {}

    def get_model_input_time(t_continuous: torch.Tensor) -> torch.Tensor:
        """Convert continuous time to model input time."""
        if noise_schedule.schedule == "discrete_flow":
            return t_continuous * noise_schedule.total_N
        return t_continuous

    def noise_pred_fn(
        x: torch.Tensor, 
        t_continuous: torch.Tensor, 
        cond: Optional[torch.Tensor] = None,
        **extra_kwargs,
    ) -> torch.Tensor:
        """Predict noise from the model output."""
        t_input = get_model_input_time(t_continuous)
        
        # Merge model_kwargs with extra_kwargs (extra_kwargs takes precedence)
        merged_kwargs = {**model_kwargs, **extra_kwargs}
        
        if cond is None:
            output = model(x, t_input, **merged_kwargs)
        else:
            output = model(x, t_input, cond, **merged_kwargs)
        
        if model_type == "flow":
            # Flow model predicts velocity v = noise - x_0
            # noise = (1 - sigma_t) * v + x
            _, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            sigma_t_expanded = expand_dims(sigma_t, x.ndim - sigma_t.ndim + 1).to(x)
            noise = (1 - sigma_t_expanded) * output + x
            return noise
        elif model_type == "noise":
            return output
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def model_fn(x: torch.Tensor, t_continuous: torch.Tensor) -> torch.Tensor:
        """The noise prediction model function for DPM-Solver."""
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            
            # CFG: combine conditional and unconditional predictions
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t_continuous] * 2)
            c_in = torch.cat([unconditional_condition, condition])
            
            # Expand model_kwargs tensors for CFG (double batch size)
            cfg_kwargs = {}
            for k, v in model_kwargs.items():
                if isinstance(v, torch.Tensor):
                    cfg_kwargs[k] = torch.cat([v] * 2)
                else:
                    cfg_kwargs[k] = v
            
            noise_output = noise_pred_fn(x_in, t_in, cond=c_in, **cfg_kwargs)
            noise_uncond, noise_cond = noise_output.chunk(2)
            
            return noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        
        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")

    return model_fn


class DPM_Solver:
    """DPM-Solver for fast sampling of diffusion models.
    
    This implements the DPM-Solver++ algorithm optimized for Flow Matching models.
    It uses higher-order ODE solvers to achieve the same quality with fewer steps.
    
    Reference: https://arxiv.org/abs/2211.01095
    """
    
    def __init__(
        self,
        model_fn: Callable,
        noise_schedule: NoiseScheduleFlow,
        algorithm_type: str = "dpmsolver++",
    ):
        """Initialize the DPM-Solver.
        
        Args:
            model_fn: Wrapped model function from model_wrapper().
            noise_schedule: Noise schedule object.
            algorithm_type: Either "dpmsolver" or "dpmsolver++".
        """
        def _expand_time(x, t):
            if t.ndim == 1:
                return t.expand(x.shape[0])
            return t.expand(x.shape[0], *t.shape[1:])

        self.model = lambda x, t: model_fn(x, _expand_time(x, t))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type

    def noise_prediction_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the noise prediction."""
        return self.model(x, t)

    def data_prediction_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the data prediction (x0)."""
        noise = self.noise_prediction_fn(x, t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        sigma_t = self.noise_schedule.marginal_std(t)
        
        sigma_t = expand_dims(sigma_t, x.ndim - sigma_t.ndim + 1)
        alpha_t = expand_dims(alpha_t, x.ndim - alpha_t.ndim + 1)
        
        x0 = (x - sigma_t * noise) / alpha_t
        return x0

    def model_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert the model to noise or data prediction based on algorithm type."""
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        return self.noise_prediction_fn(x, t)

    def get_time_steps(
        self, 
        skip_type: str, 
        t_T: float, 
        t_0: float, 
        N: int, 
        device: torch.device,
        shift: float = 1.0,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """Compute the intermediate time steps for sampling.
        
        Args:
            skip_type: Type of time step spacing.
            t_T: Starting time (usually 1.0).
            t_0: Ending time (usually ~0.001).
            N: Number of steps.
            device: Target device.
            shift: Flow shift parameter for time_uniform_flow.
            dtype: Data type for the timesteps tensor.
        
        Returns:
            Tensor of time steps with shape (N + 1,).
        """
        if skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1, dtype=dtype, device=device)
        elif skip_type == "time_uniform_flow":
            betas = torch.linspace(t_T, t_0, N + 1, dtype=dtype, device=device)
            sigmas = 1.0 - betas
            sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(dims=[0])
            return sigmas
        elif skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T, dtype=dtype, device=device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0, dtype=dtype, device=device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1, dtype=dtype, device=device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        else:
            raise ValueError(f"Unsupported skip_type: {skip_type}")

    def dpm_solver_first_update(
        self, 
        x: torch.Tensor, 
        s: torch.Tensor, 
        t: torch.Tensor, 
        model_s: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`."""
        ns = self.noise_schedule
        
        t = expand_dims(t, x.ndim - t.ndim + 1)
        s = expand_dims(s, x.ndim - s.ndim + 1)
        
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
        else:
            log_alpha_s = ns.marginal_log_mean_coeff(s)
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = torch.exp(log_alpha_t - log_alpha_s) * x - (sigma_t * phi_1) * model_s
        
        return x_t

    def multistep_dpm_solver_second_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
        solver_type: str = "dpmsolver"
    ) -> torch.Tensor:
        """Multistep DPM-Solver-2 from time `t_prev_list[-1]` to time `t`."""
        t = expand_dims(t, x.ndim - t.ndim + 1)
        ns = self.noise_schedule
        
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        
        t_prev_1 = expand_dims(t_prev_1, x.ndim - t_prev_1.ndim + 1)
        t_prev_0 = expand_dims(t_prev_0, x.ndim - t_prev_0.ndim + 1)
        
        lambda_prev_1 = ns.marginal_lambda(t_prev_1)
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        
        log_alpha_prev_0 = ns.marginal_log_mean_coeff(t_prev_0)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == "dpmsolver":
                x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
            else:
                x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 + (alpha_t * (phi_1 / h + 1.0)) * D1_0
        else:
            phi_1 = torch.expm1(h)
            if solver_type == "dpmsolver":
                x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - (sigma_t * phi_1) * model_prev_0 - 0.5 * (sigma_t * phi_1) * D1_0
            else:
                x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - (sigma_t * phi_1) * model_prev_0 - (sigma_t * (phi_1 / h - 1.0)) * D1_0
        
        return x_t

    def multistep_dpm_solver_update(
        self,
        x: torch.Tensor,
        model_prev_list: List[torch.Tensor],
        t_prev_list: List[torch.Tensor],
        t: torch.Tensor,
        order: int,
        solver_type: str = "dpmsolver"
    ) -> torch.Tensor:
        """Multistep DPM-Solver with the order `order`."""
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError(f"Solver order must be 1 or 2, got {order}")

    def sample(
        self,
        x: torch.Tensor,
        steps: int = 20,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        order: int = 2,
        skip_type: str = "time_uniform_flow",
        method: str = "multistep",
        lower_order_final: bool = True,
        solver_type: str = "dpmsolver",
        flow_shift: float = 3.0,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        """Sample from the diffusion model using DPM-Solver.
        
        Args:
            x: Initial noise tensor.
            steps: Number of sampling steps.
            t_start: Starting time (default: 1.0).
            t_end: Ending time (default: 0.001).
            order: Order of the solver (1 or 2).
            skip_type: Time step spacing type.
            method: Sampling method ("multistep").
            lower_order_final: Use lower order at final steps.
            solver_type: Type of solver ("dpmsolver" or "taylor").
            flow_shift: Flow shift parameter.
            return_intermediate: Return intermediate results.
        
        Returns:
            Sampled tensor (denoised result).
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0"
        
        device = x.device
        dtype = x.dtype
        intermediates = []
        
        with torch.no_grad():
            if method == "multistep":
                assert steps >= order
                timesteps = self.get_time_steps(
                    skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device, shift=flow_shift, dtype=dtype
                )
                assert timesteps.shape[0] - 1 == steps
                
                # Initialize
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                
                if return_intermediate:
                    intermediates.append(x)
                
                # Initialize first `order` values by lower order multistep solver
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step, solver_type=solver_type
                    )
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                
                # Compute remaining values
                disable_tqdm = os.getenv("DPM_TQDM", "False") == "True"
                for step in tqdm(range(order, steps + 1), disable=disable_tqdm, desc="DPM-Solver"):
                    t = timesteps[step]
                    
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type
                    )
                    
                    if return_intermediate:
                        intermediates.append(x)
                    
                    # Update history
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    
                    # Don't need to evaluate model at final step
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            else:
                raise ValueError(f"Unsupported method: {method}")
        
        if return_intermediate:
            return x, intermediates
        return x


def create_flow_dpm_solver(
    model: Callable,
    condition: torch.Tensor,
    uncondition: torch.Tensor,
    cfg_scale: float,
    model_kwargs: Optional[Dict] = None,
) -> DPM_Solver:
    """Create a Flow-DPM-Solver for sampling.
    
    This is a convenience function that creates a DPM-Solver configured
    for Flow Matching models with classifier-free guidance.
    
    Args:
        model: The diffusion model's forward function.
        condition: Conditional text embeddings.
        uncondition: Unconditional text embeddings.
        cfg_scale: Classifier-free guidance scale.
        model_kwargs: Additional model kwargs.
    
    Returns:
        Configured DPM_Solver instance.
    
    Example:
        >>> solver = create_flow_dpm_solver(model.forward_with_dpmsolver, 
        ...     text_embeds, uncond_embeds, cfg_scale=4.5)
        >>> latents = solver.sample(noise, steps=20, flow_shift=3.0)
    """
    noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")
    
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="flow",
        model_kwargs=model_kwargs or {},
        guidance_type="classifier-free",
        condition=condition,
        unconditional_condition=uncondition,
        guidance_scale=cfg_scale,
    )
    
    return DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

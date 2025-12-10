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
Inference utilities for Muse models.

This module provides sampling and inference tools including:
- Flow-DPM-Solver for accelerated sampling of Flow Matching models
"""

from muse.inference.flow_dpm_solver import (
    DPM_Solver,
    NoiseScheduleFlow,
    create_flow_dpm_solver,
    model_wrapper,
)

__all__ = [
    "DPM_Solver",
    "NoiseScheduleFlow",
    "create_flow_dpm_solver",
    "model_wrapper",
]

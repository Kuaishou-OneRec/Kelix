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
Sana inference utilities.

This module provides sampling and inference tools for Sana models.
"""

from muse.inference.sana.sampling import (
    decode_latents,
    encode_prompts,
    generate_with_dpm_solver,
    generate_with_euler,
)

__all__ = [
    "encode_prompts",
    "decode_latents",
    "generate_with_dpm_solver",
    "generate_with_euler",
]

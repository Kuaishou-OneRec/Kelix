#!/usr/bin/env python

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This script extracts fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints. It gets
# copied into the top level checkpoint dir, so the user can easily do the conversion at any point in
# the future. Once extracted, the weights don't require DeepSpeed and can be used in any
# application.
#
# example: python zero_to_fp32.py . pytorch_model.bin

import argparse
import os
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def get_ds_checkpoint_dir(checkpoint_dir, tag=None):
  if tag is None:
    latest_path = os.path.join(checkpoint_dir, 'latest')
    if os.path.isfile(latest_path):
      with open(latest_path, 'r') as fd:
        tag = fd.read().strip()
    else:
      raise ValueError(f"Unable to find 'latest' file at {latest_path}")

  ds_checkpoint_dir = os.path.join(checkpoint_dir, tag)

  if not os.path.isdir(ds_checkpoint_dir):
    raise FileNotFoundError(f"Directory '{ds_checkpoint_dir}' doesn't exist")

  return ds_checkpoint_dir


def convert_zero_checkpoint_to_state_dict(
        checkpoint_dir,
        output_file="pytorch_model.bin",
        tag=None,
        dtype="fp32"):
  """
  Convert ZeRO 2 or 3 checkpoint into a single fp32 consolidated ``state_dict`` file that can be
  loaded with ``torch.load(file)`` + ``load_state_dict()`` and used for training without DeepSpeed.

  Args:
      - ``checkpoint_dir``: path to the desired checkpoint folder. (one that contains the tag-folder, like ``global_step14``)
      - ``output_file``: path to the pytorch fp32 state_dict output file (e.g. path/pytorch_model.bin)
      - ``tag``: checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named ``latest`` in the checkpoint folder, e.g., ``global_step14``
  """
  if dtype == "fp32":
    dtype_ = torch.float32
  elif dtype == "fp16":
    dtype_ = torch.float16
  elif dtype == "bf16":
    dtype_ = torch.bfloat16
  else:
    raise ValueError(f"Unsupported dtype {dtype}")

  state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
  for key in list(state_dict.keys()):
    state_dict[key] = state_dict[key].to(dtype_)

  ds_checkpoint_dir = get_ds_checkpoint_dir(checkpoint_dir, tag)
  os.makedirs(
      os.path.join(ds_checkpoint_dir, dtype), exist_ok=True)
  output_path = os.path.join(ds_checkpoint_dir, dtype, output_file)
  print(f"Saving {dtype} state dict to {output_path}")
  torch.save(state_dict, output_path)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "checkpoint_dir",
      type=str,
      help="path to the desired checkpoint folder, e.g., path/checkpoint-12")

  parser.add_argument(
      "--output_file",
      type=str,
      default="pytorch_model.bin",
      help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)")

  parser.add_argument(
      "--tag",
      type=str,
      default=None,
      help="checkpoint tag used as a unique identifier for checkpoint. If not provided will attempt to load tag in the file named 'latest' in the checkpoint folder, e.g., global_step14")

  parser.add_argument(
      "--dtype",
      type=str,
      default="fp32",
      choices=["fp32", "fp16", "bf16"],
      help="dtype of the output state_dict")

  args = parser.parse_args()

  convert_zero_checkpoint_to_state_dict(
      args.checkpoint_dir, args.output_file, args.tag, args.dtype)

from typing import Union, Dict

import os
import json
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_file
import tqdm
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.default_planner import (
    _EmptyStateDictLoadPlanner
)
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import re
from recovlm.utils.ds_utils import print_input_info

# cd /llm_reco/lingzhixin/recovlm_qw0510/recovlm; PYTHONPATH=. python3 tools/model_helpers/slowfast/inspect_distcp.py /mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.1/Stage3_SlowFast/8b/slowfast_0723/step71000/global_step71000 > recovlm.out 2>&1 &;
import sys
dcp_checkpoint_dir = sys.argv[1]

if dcp_checkpoint_dir.endswith(".pt"):
    sd = torch.load(dcp_checkpoint_dir)
else:
    sd: STATE_DICT_TYPE = {}
    import os
    print(f"dcp_to_torch_save({os.getpid()}): _load_state_dict ...")
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

print(f"sd length = {len(sd)}")
print_input_info(
    sd,
    "sddddd"
)

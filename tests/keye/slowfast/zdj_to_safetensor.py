import os
import json
import torch
import argparse
import os.path as osp
from safetensors.torch import save_file


def main(args):
    if args.output.strip() == "":
        args.output = args.folder
    pth_file = osp.join(args.folder, "mp_rank_00_model_states.pt")
    model = torch.load(pth_file, map_location='cpu')["module"]
    model = {
        name.replace("siglip", args.prefix): model[name]
        for name in model.keys()
        if "vision_model" in name
    }
    os.makedirs(args.output, exist_ok=True)
    if args.style == "pt":
        torch.save(model, osp.join(args.output, "vision_model.pth"))
    else:
        save_file(model, osp.join(args.output, "vision_model.safetensors"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--output', type=str, default="")
    parser.add_argument('--prefix', type=str, default="visual")
    parser.add_argument('--style', type=str, choices=["pt", "safetensors"], default="pt")
    ags = parser.parse_args()
    main(ags)



"""
python3 tests/keye/slowfast/zdj_to_safetensor.py --folder /mmu_mllm_hdd_2/zhouyang12/output2/Keye/0.8.0/ViT/80m/0.0.1/global_step19800 --style safetensors
"""
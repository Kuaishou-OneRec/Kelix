import os
import os.path as osp
import glob
import argparse
import torch
import shutil


config_files_dir = "./model_conf"


def get_input_model_file(input_dir):
    return osp.join(input_dir, 'mp_rank_00_model_states.pt')

def copy_config_files(output_dir):
    source_files = glob.glob(osp.join(config_files_dir, '*'))
    source_files = [
        file
        for file in source_files
        if osp.isfile(file) and osp.basename(file) not in ["pytorch_model.bin", "tmp.txt"]
    ]
    for source_file in source_files:
        dest_file = osp.join(output_dir, osp.basename(source_file))
        if osp.exists(dest_file):
            continue
        print("Copy", source_file)
        shutil.copy(source_file, dest_file)


def main(args):
    input_pt = get_input_model_file(args.input_dir)
    model = torch.load(input_pt, map_location="cpu")
    module = model["module"]
    os.makedirs(args.output_dir, exist_ok=True)
    output_pt = osp.join(args.output_dir, 'pytorch_model.bin')
    torch.save(module, output_pt)
    copy_config_files(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    ags = parser.parse_args()
    main(ags)
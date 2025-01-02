import os
import argparse
import torch
import webdataset as wds
from omegaconf import OmegaConf

def setup_proc(args):
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', "0"))
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', "1"))
    master_addr, master_port = args.master_addr.split(":")
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ["LOCAL_RANK"] = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', "0")
    os.environ["WORLD_SIZE"] = os.environ.get('OMPI_COMM_WORLD_SIZE', "1")
    os.environ["RANK"] = os.environ.get('OMPI_COMM_WORLD_RANK', "0")

class Worker(object):

    def __init__(self, config):
        self.config = config
    
    def run(self):
        pass

def main():
    parser = argparser.ArgumentParser(description="clean data")
    parser.add_argument("config_file")
    parser.add_argument(
        "--master-addr",
        type=str,
        required=True)
    args = parser.parse_args()

if __name__ == '__main__':
    main()

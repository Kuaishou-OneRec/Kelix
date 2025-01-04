import os
import json
import argparse
import pandas as pd
from glob import glob
from mpi4py import MPI
import pandas as pd
import webdataset as wds
from tqdm import tqdm
from omegaconf import OmegaConf
from worker import MPITarWriterWorker
from datasets import create_dataset
from converters import create_converter

class DatasetConverterWorker(MPITarWriterWorker):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dataset = create_dataset(config.dataset)
        self.converter = create_converter(config.converter)

    def run(self):
        for s in tqdm(self.dataset):
            try:
                out = self.converter(s)
                self.write_sample(out)
            except Exception as e:
                print(e)

def main():
    parser = argparse.ArgumentParser(
        description="tool for convert dataset to recovlm webdataset"
    )
    parser.add_argument("config_file")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_file)
    print("config", cfg)
    worker = DatasetConverterWorker(cfg)
    worker.run()


if __name__ == "__main__":
    main()
import os
import json
import argparse
import traceback
from glob import glob
from mpi4py import MPI
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from worker import MPIBase
from datasets import create_dataset
from converters import create_converter
from filters import create_filter
import gc

class JsonDataConverterWorker(MPIBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dir = config.output_dir
        
        # 确保输出目录存在
        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
        self.comm.barrier()
        
        # 初始化数据集和转换器
        self.dataset = create_dataset(config.dataset)
        
        # 初始化转换器列表
        self._converters = []
        if "converter" in config:
            self._converters.append(create_converter(config.converter))
        if "converters" in config:
            for cfg in config['converters']:
                self._converters.append(create_converter(cfg))
                
        # 初始化过滤器
        self._pre_filters = []
        if "pre_filters" in config:
            for cfg in config['pre_filters']:
                self._pre_filters.append(self.create_filter(cfg))
        
        self._post_filters = []
        if "post_filters" in config:
            for cfg in config['post_filters']:
                self._post_filters.append(self.create_filter(cfg))
        
        # 统计计数器
        self._filtered_cnt = 0
        self._success_cnt = 0
        self._none_cnt = 0
        self._filter_reason = dict()
        
        # 为每个worker创建单独的输出文件
        self.output_file = os.path.join(
            self.output_dir, 
            f"wenjuan_data_{self.rank:05d}.json"
        )
        
        # 清空或删除已存在的输出文件
        try:
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
                self.mpi_print(f"Removed existing output file: {self.output_file}")
        except Exception as e:
            self.mpi_print(f"Error removing existing output file: {str(e)}")
            raise

        # 如果需要合并文件，也清理合并后的文件
        if self.rank == 0 and self.config.get('merge_output', False):
            merged_file = os.path.join(self.output_dir, "wenjuan_data_merged.json")
            try:
                if os.path.exists(merged_file):
                    os.remove(merged_file)
                    self.mpi_print(f"Removed existing merged file: {merged_file}")
            except Exception as e:
                self.mpi_print(f"Error removing existing merged file: {str(e)}")
                raise

        self.comm.barrier()  # 确保所有worker都完成了清理工作

    def create_filter(self, cfg):
        return self.filter_wrapper(cfg.class_name, create_filter(cfg))
    
    def filter_wrapper(self, name, filter_func):
        def f(x):
            rst = filter_func(x)
            if not rst:
                self._filter_reason[name] = self._filter_reason.get(name, 0) + 1
            return rst
        return f

    def write_sample(self, sample):
        """将样本写入JSON文件，确保UTF-8编码"""
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                # json.dumps() 不需要 encoding 参数
                json_str = json.dumps(sample, ensure_ascii=False)
                f.write(json_str + '\n')
            self._success_cnt += 1
            
            if self._success_cnt % 1000 == 0:
                self.mpi_print(f"Processed {self._success_cnt} samples")
                self.mpi_print(f"Filter reason: {self._filter_reason}")
        except Exception as e:
            self.mpi_print(f"Error writing sample: {str(e)}")

    def run(self):
        """运行转换过程"""
        self.mpi_print(f"Starting conversion, writing to {self.output_file}")
        
        # 获取并打印数据集总数
        total_rows = None if not hasattr(self.dataset, "__len__") else len(self.dataset)
        
        try:
            for sample in tqdm(self.dataset, total=total_rows):
                try:
                    # 应用前置过滤器
                    if not all([f(sample) for f in self._pre_filters]):
                        self._filtered_cnt += 1
                        continue
                        
                    # 应用所有转换器
                    out = sample
                    for converter in self._converters:
                        out = converter(out)
                        
                    # 应用后置过滤器
                    if not all([f(out) for f in self._post_filters]):
                        self._filtered_cnt += 1
                        continue
                        
                    # 写入输出
                    if out is not None:
                        if isinstance(out, dict):
                            self.write_sample(out)
                        elif isinstance(out, (list, tuple)):
                            for s in out:
                                self.write_sample(s)
                    else:
                        self._none_cnt += 1
                        
                except Exception as e:
                    self.mpi_print(f"Error processing sample: {str(e)}")
                    self.mpi_print(traceback.format_exc())
                    continue
                    
            self.mpi_print(f"Finished processing. Success: {self._success_cnt}, "
                          f"Filtered: {self._filtered_cnt}, None: {self._none_cnt}")
            
        except Exception as e:
            self.mpi_print(f"Error in run: {str(e)}")
            self.mpi_print(traceback.format_exc())
        finally:
            self.comm.barrier()
            gc.collect()
            
            # 在rank 0上合并所有文件（可选）
            if self.rank == 0 and self.config.get('merge_output', False):
                self._merge_output_files()

    def _merge_output_files(self):
        """合并所有worker的输出文件，确保UTF-8编码"""
        try:
            merged_file = os.path.join(self.output_dir, "wenjuan_data_merged.json")
            self.mpi_print(f"Merging output files to {merged_file}")
            
            with open(merged_file, 'w', encoding='utf-8') as outfile:
                for rank in range(self.world_size):
                    rank_file = os.path.join(
                        self.output_dir,
                        f"wenjuan_data_{rank:05d}.json"
                    )
                    if os.path.exists(rank_file):
                        with open(rank_file, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                outfile.write(line)
                                
            self.mpi_print("Merge completed")
            
        except Exception as e:
            self.mpi_print(f"Error merging files: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Tool for converting dataset to JSON format"
    )
    parser.add_argument("config_file", help="Path to the configuration file")
    args = parser.parse_args()
    
    # 加载配置
    cfg = OmegaConf.load(args.config_file)
    print("Config:", cfg)
    
    # 创建worker并运行
    worker = JsonDataConverterWorker(cfg)
    worker.run()

if __name__ == "__main__":
    main()

"""Batch inference for Qwen3Siglip"""
from absl import flags, app
import json
import collections
import os
import sys
import logging
import torch
import pandas as pd
from typing import Dict, List
from recovlm.models.qwen3siglip.modeling_qwen3siglip import Qwen3SiglipForConditionalGeneration_navit
from recovlm.models.qwen3siglip.processing_qwen3siglip import Qwen3SiglipProcessor_siglip
import math
from msy_infer_dataset import MsyInferDataset
import pyarrow.parquet as pq
from recovlm.training.common import set_default_dtype, get_global_grad_norm, clip_grad_by_value
import time
# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 全局变量用于存储 MPI 模块
MPI = None

def init_mpi():
    """初始化 MPI 环境"""
    global MPI
    try:
        from mpi4py import MPI as mpi_module
        MPI = mpi_module
        
        # 设置NCCL环境变量以提高容错性
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_TIMEOUT"] = "1800"  # 30分钟超时
        os.environ["NCCL_DEBUG"] = "INFO"
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        hostname = MPI.Get_processor_name()
        return comm, rank, size, hostname
    except ImportError:
        raise ImportError(
            "mpi4py is required for distributed inference. "
            "Please install it with: pip install mpi4py"
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize MPI environment: {str(e)}. "
            "Please make sure mpi4py is properly installed and "
            "the script is launched with mpirun."
        )

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from tqdm import tqdm
from vllm import LLM, SamplingParams
from wenjuan_infer_dataset import WenJuanInferDataset
from torch.utils.data import DataLoader


FLAGS = flags.FLAGS

flags.DEFINE_string(
  "model_name_or_path", "/llm_reco_ssd/zhouyang12/models/Qwen3-1.7B-siglip", "The path or name of model."
)

flags.DEFINE_string(
  "parquet_path", "/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_11w_cot_v2/photo_0210_11w_sft_data-test.parquet", "The path or name of model."
)

flags.DEFINE_integer(
  "tp", 1, "The tensor_parallel_size"
)

flags.DEFINE_string(
  "output_path", "msy_test", "The path of file to write results." 
)

flags.DEFINE_integer(
  "batch_size", 1, "The batch size for inference."
)

flags.DEFINE_string(
  "user", "mpi", "The HDFS user name when reading from HDFS."
)

flags.DEFINE_integer(
    "global_rank", 0,
    "Global rank for multiple MPI jobs running in parallel"
)


def collate_fn(samples):
  batch = collections.defaultdict(list)
  for sample in samples:
    for key, item in sample.items():
      batch[key].append(item)
  return batch

def split_dataset(dataset, comm_size, rank):
    """Split dataset for MPI processes using IterableDataset approach"""
    class RankedIterableDataset(torch.utils.data.IterableDataset):
        def __init__(self, dataset, rank, world_size):
            super().__init__()
            self.dataset = dataset
            self.rank = rank
            self.world_size = world_size

        def __iter__(self):
            count = 0
            for item in self.dataset:
                if count % self.world_size == self.rank:
                    yield item
                count += 1

        def __len__(self):
            return len(self.dataset) // self.world_size + (1 if len(self.dataset) % self.world_size > self.rank else 0)

    return RankedIterableDataset(dataset, rank, comm_size)

def merge_results(local_results_path, comm, rank, output_path, global_rank, dataset_name):
    """Merge results from all MPI processes with error handling"""
    try:
        if rank == 0:
            all_results = []
            total_ppl_sum = 0.0
            total_count = 0
            
            # 读取主进程的结果
            with open(local_results_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    if "total_ppl" in result:
                        total_ppl_sum += result["total_ppl"]
                        total_count += result.get("count", 1)
                    all_results.append(line.strip())
            
            # 从其他进程收集结果
            for i in range(1, comm.Get_size()):
                try:
                    worker_results = comm.recv(source=i, tag=11, status=MPI.Status())
                    if worker_results:
                        for result_str in worker_results:
                            result = json.loads(result_str)
                            if "total_ppl" in result:
                                total_count += result["count"]
                                total_ppl_sum = (total_count-result["count"])/total_count *total_ppl_sum +result["total_ppl"]*result["count"]/total_count
                            all_results.append(result_str)
                except Exception as e:
                    logging.error(f"Error receiving results from rank {i}: {e}")
                    continue
            logging.info(f"Average PPL across all ranks: {total_ppl_sum:.4f}")
            
            # 写入合并后的结果
            final_output_path = f"{output_path}_{dataset_name}.jsonl"
            with open(final_output_path, 'w', encoding='utf-8') as f:
                # 首先写入平均PPL
                f.write(json.dumps({"average_ppl": total_ppl_sum, "total_samples": total_count}, ensure_ascii=False) + "\n")
                # 然后写入所有详细结果
                for result in all_results:
                    f.write(result + '\n')
        else:
            try:
                # 工作进程发送结果
                with open(local_results_path, 'r', encoding='utf-8') as f:
                    results = [line.strip() for line in f]
                comm.send(results, dest=0, tag=11)
            except Exception as e:
                logging.error(f"Error sending results from rank {rank}: {e}")
                comm.send(None, dest=0, tag=11)  # 发送空结果表示错误
    except Exception as e:
        logging.error(f"Error in merge_results on rank {rank}: {e}")

def parse_hostfile(hostfile_path):
    """解析 hostfile 获取机器数量和每台机器的 slots"""
    try:
        with open(hostfile_path, 'r') as f:
            hosts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        total_machines = len(hosts)
        # 解析每行获取 slots 信息，格式可能是 "hostname slots=N" 或纯 "hostname"
        slots_per_machine = []
        machines = []
        for host in hosts:
            parts = host.split()
            if len(parts) > 1 and 'slots=' in parts[1]:
                slots = int(parts[1].split('=')[1])
            else:
                slots = 1  # 默认值
            slots_per_machine.append(slots)
            machines.append(host)
            
        return total_machines, slots_per_machine, machines
    except Exception as e:
        raise RuntimeError(f"Failed to parse hostfile {hostfile_path}: {str(e)}")

def main(_):
    comm, rank, size, hostname = init_mpi()
    print('rank:', rank)
    print('size:', size)
    torch.cuda.set_device(rank)
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU to run.")
    device_count = torch.cuda.device_count()
    logging.info(f"Found {device_count} CUDA devices")
    for i in range(device_count):
        logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # 初始化 MPI
    
    # 从环境变量获取 hostfile 路径，如果没有则使用命令行参数
    hostfile = os.getenv('HOSTFILE')
    if not hostfile:
        # 尝试从 MPI 获取 hostfile 路径
        try:
            info = MPI.Info.Create()
            hostfile = info.Get('hostfile')
        except:
            # 如果都获取不到，使用默认值或报错
            if rank == 0:
                logging.warning("Could not get hostfile path from environment or MPI info. "
                              "Proceeding without hostfile validation...")
            hostfile = None
    
    # 只在有 hostfile 时进行验证
    if hostfile and os.path.exists(hostfile):
        total_machines, slots_per_machine, machines = parse_hostfile(hostfile)
        total_slots = sum(slots_per_machine)
        
        # 详细的配置验证
        if rank == 0:
            logging.info("=== Cluster Configuration ===")
            logging.info(f"Hostfile: {hostfile}")
            logging.info(f"Total machines: {total_machines}")
            logging.info(f"Machines and slots:")
            for i, (machine, slots) in enumerate(zip(machines, slots_per_machine)):
                logging.info(f"  {i+1}. {machine}: {slots} slots")
            logging.info(f"Total slots: {total_slots}")
            logging.info(f"Required processes: {size}")
            logging.info(f"Tensor parallel size: {FLAGS.tp}")
            logging.info("=" * 30)
        
        # 验证配置
        if size > total_slots:
            raise ValueError(
                f"MPI process count ({size}) exceeds available slots in hostfile ({total_slots})"
            )
    else:
        if rank == 0:
            logging.warning("Running without hostfile validation. "
                          "Make sure your MPI configuration is correct!")
    
    # 确保所有进程都正确初始化
    comm.Barrier()
    
    # 初始化采样参数
    # Load dataset
    time_start = time.time()
    datasetlist = {
        # "MMBench":"/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/en/dev-00000-of-00001.parquet",
        # "MMBenchCn":"/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMBench/cn/dev-00000-of-00001.parquet",
        # "MME":"/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MME/MME.parquet",
        "MMTBench":"/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMTBench/mmt_bench_485_hetu_format.parquet"
        # "MMStar":"/llm_reco/maosiyang/dataset/mmstar/MMStar/mmstar.parquet",
        # "MathVista":"/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MathVista/mathvista.parquet",
        # "OCRBench":"/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/OCRBench/data/test-00000-of-00001.parquet",
        # "Benchmark_v21":"/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/Benchmark_v21/Benchmark_v21.parquet",
        # "AI2D":"/llm_reco_ssd/luoxinchen/dataset/ai2d/ai2d/data/merge/test-00000-of-00001.parquet",
        # "AI2D_no_mask":"/llm_reco_ssd/luoxinchen/dataset/ai2d/ai2d-no-mask/data/merge/test-00000-of-00001.parquet",
        # "infoVQA":"/llm_reco_ssd/luoxinchen/dataset/infoVQA/human_download/infographicsvqa_qas/reconstruct_val.parquet",
        # "RealWorldQA":"/llm_reco_ssd/luoxinchen/dataset/RealWorldQA/RealWorldQA/data/merge/test-00000-of-00001.parquet"
    }
    with set_default_dtype(torch.bfloat16):
        llm = Qwen3SiglipForConditionalGeneration_navit.from_pretrained(
            FLAGS.model_name_or_path,
            _attn_implementation = 'flash_attention_2',
            use_cache=False
        )
        # llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        # )
    llm = llm.to(torch.cuda.current_device())
    # Split dataset for this MPI rank
    for dataset_name, dataset_path in datasetlist.items():
        dataset = MsyInferDataset(dataset_name=dataset_name, parquet_path=dataset_path, model_name_or_path=FLAGS.model_name_or_path, user='mpi')
        local_dataset = split_dataset(dataset, size, rank)
        # Create local results file for this rank using both local and global rank
        local_output_path = f"{FLAGS.output_path}_{dataset_name}.jsonl.rank{rank}"

        # Process local chunk of data
        with open(local_output_path, "w", encoding="utf-8") as f:
            count = 1
            total_ppl = 0
            for batch in tqdm(DataLoader(local_dataset,batch_size=FLAGS.batch_size,
                                    collate_fn=collate_fn),disable=rank != 0):  # Only rank 0 shows progress bar
                # 存储该批次所有样本的所有生成结果
                batch_generations = [[] for _ in range(len(batch["inputs"]))]
                with torch.no_grad():
                    for idx in range(len(batch["inputs"])):
                        answer_idx_list = batch["answer_idx_list"][idx]
                        inputs = batch["inputs"][idx].to(torch.cuda.current_device())
                        input_ids = inputs["input_ids"]
                        with torch.no_grad():
                            outputs = llm(**inputs)
                            logits = outputs.logits 
                        start_pos, end_pos = answer_idx_list[0]
                        try:
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = input_ids[..., 1:].contiguous()

                            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            loss = loss.view(shift_logits.size(0), -1)

                            assistant_loss = loss[0, start_pos-1:end_pos]
                            response_ppl = torch.exp(assistant_loss.mean())
                        except Exception as e:
                            logging.warning(f"Error calculating PPL for position {start_pos}: {e}")
                            continue
                    
                        total_ppl = response_ppl/count+total_ppl*(count-1)/count
                        count += 1
                        print('response_ppl:', response_ppl)
            print('==================================================')
            # 先将tensor转换为float32，再转换为numpy数组
            total_ppl = float(total_ppl.to(torch.float32).cpu().numpy())
            print('total_ppl:', total_ppl, 'rank:', rank)
            result = {
                "total_ppl": total_ppl,
                "count": count-1,
                "rank": rank
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
        comm.Barrier()
        merge_results(local_output_path, comm, rank, FLAGS.output_path, FLAGS.global_rank, dataset_name)
        if rank == 0:
            # Merge results from all processes
            logging.info(f"Results being written to: {FLAGS.output_path}_{dataset_name}.jsonl")
            for r in range(size):
                temp_file = f"{FLAGS.output_path}_{dataset_name}.jsonl.rank{r}"
                if os.path.exists(temp_file):
                    os.remove(temp_file)

if __name__ == "__main__":
    app.run(main)
    time_end = time.time()
    logging.info(f"Total time: {time_end - time_start:.2f} seconds")
import os
import glob
import json
import uuid
import torch
import traceback
import argparse
import time
from datetime import datetime
from omegaconf import OmegaConf
from transformers import AutoProcessor
from torch.utils.data import DataLoader, Dataset
from model import load_vllm_engine, infer
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

class VideoQADataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        # 所有数据（视频和图片）都在这个路径下
        self.base_path = "/llm_reco/chuchenglong/R3/Ai_kwai/video/e49f69da289f4d02bebe48cf6a033a2c"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            pid = item["pid"]
            query = item["query"]
            video_info = item["video_info"]
            
            # 检查是否有匹配 pid_*.jpg 的图像
            image_pattern = os.path.join(self.base_path, f"{pid}_*.jpg")
            image_paths = glob.glob(image_pattern)
            
            if image_paths:
                # 按照数字排序图像
                def extract_frame_number(path):
                    try:
                        return int(path.split('_')[-1].split('.')[0])
                    except (IndexError, ValueError):
                        return float('inf')
                
                image_paths = sorted(image_paths, key=extract_frame_number)
                sample = {
                    'pid': pid,
                    'query': query,
                    'video_info': video_info,
                    'image_paths': image_paths  # 直接使用找到的图像路径
                }
                return sample
            
            # 如果找不到图像，尝试寻找视频文件
            for ext in ['.mp4', '.mkv', '.avi', '.mov']:
                video_path = os.path.join(self.base_path, f"{pid}{ext}")
                if os.path.exists(video_path):
                    sample = {
                        'pid': pid,
                        'query': query,
                        'video_info': video_info,
                        'video_path': video_path  # 使用找到的视频文件
                    }
                    return sample
            
            # 如果两种路径都找不到，返回None
            print(f"找不到pid: {pid}的视频或图像")
            return None
        except Exception as e:
            print(f"处理数据索引 {idx} 时出错: {str(e)}")
            return None

class VideoQACollator:
    def __init__(self, processor):
        self.processor = processor
        
    def _process_single(self, sample):
        if sample is None:
            return None, None
            
        pid = sample["pid"]
        query = sample["query"]
        video_info = sample["video_info"]
        
        # 获取图片路径和视频路径
        image_paths = sample.get("image_paths", [])
        video_path = sample.get("video_path", None)
        
        # 构建提示
        prompt = f"以下是一个关于视频的问题，请根据视频内容回答：\n问题：{query}\n视频信息：{video_info}"
        
        # 构建用户内容
        user_content = []
        
        # 处理内容 - 区分图片序列和视频文件
        if image_paths:
            # 如果有图片序列，将其作为视频帧处理
            user_content.append({
                "type": "video",
                "video": image_paths,  # 直接传递图片路径列表
                "fps": 1.0,
            })
        elif video_path:
            # 如果有视频文件，直接使用视频路径
            user_content.append({
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,
                "fps": 1.0,
            })
        else:
            # 如果既没有图片也没有视频，返回None
            return None, None
        
        # 添加文本提示
        user_content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "system", "content": "你是一个视频问答专家，根据视频内容和提供的信息以评论的形式回答用户问题。"},
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # 使用处理器应用模板
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理视觉信息 - 需要使用适合视频的处理方法
        image_inputs, video_inputs , *extra = process_vision_info(messages, return_video_kwargs=True)
        
        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        elif image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        raw_inputs = {
            "pid": pid,
            "query": query,
            "video_info": video_info,
            "video_path": video_path if video_path else None,
            "image_paths": image_paths if image_paths else None
        }

        return raw_inputs, llm_inputs
    
    def __call__(self, samples):
        raw_inputs_list = []
        llm_inputs_list = []
        
        for sample in samples:
            try:
                if sample is None:
                    continue
                raw_inputs, llm_inputs = self._process_single(sample)
                if llm_inputs:
                    raw_inputs_list.append(raw_inputs)
                    llm_inputs_list.append(llm_inputs)
            except Exception as e:
                print(f"处理样本错误: {str(e)}")
                traceback.print_exc()
                
        return raw_inputs_list, llm_inputs_list

class VideoQAInference:
    def __init__(self, config):
        self.config = config
        
        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 设置结果文件路径
        self.result_file = os.path.join(config.output_dir, "inference_results.json")
        
        # 加载模型和处理器
        print("正在加载模型和处理器...")
        self.processor = AutoProcessor.from_pretrained(config.model.path)
        self.model, self.sampling_params = load_vllm_engine(config)
        print("模型加载完成")
    def load_data(self):
        print(f"正在加载数据: {self.config.data_path}")
        data = list()
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
            
        if not isinstance(data, list):
            # 将JSON对象转换为列表
            data = [json.loads(line) for line in data.splitlines() if line.strip()]
            
        print(f"加载了 {len(data)} 条数据")
        return data
        
    def save_results(self, results):
        """将结果增量保存到单个JSON文件"""
        try:
            # 尝试加载现有结果
            existing_results = []
            if os.path.exists(self.result_file):
                try:
                    with open(self.result_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    print(f"已加载 {len(existing_results)} 条现有结果")
                except json.JSONDecodeError:
                    print(f"结果文件 {self.result_file} 格式无效，将创建新文件")
            
            # 合并结果
            all_results = existing_results + results
            
            # 保存全部结果
            with open(self.result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
                
            print(f"已将 {len(results)} 条新结果追加到 {self.result_file}，当前共 {len(all_results)} 条结果")
            
            # 创建备份文件
            backup_file = f"{self.result_file}.bak"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            return len(all_results)
        except Exception as e:
            print(f"保存结果失败: {str(e)}")
            traceback.print_exc()
            return 0
    
    def run(self):
        # 加载数据
        data = self.load_data()
        dataset = VideoQADataset(data)
        collator = VideoQACollator(self.processor)
        
        # 创建数据加载器
        batch_size = self.config.dataloader.batch_size
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=collator, 
            shuffle=False
        )
        
        # 开始处理
        pending_results = []
        total_batches = len(dataloader)
        print(f"开始处理 {total_batches} 批数据")
        
        start_time = time.time()
        for batch_idx, (raw_inputs, llm_inputs) in enumerate(tqdm(dataloader, desc="处理批次")):
            if not llm_inputs:
                print(f"批次 {batch_idx} 没有有效输入，跳过")
                continue
                
            try:
                # 模型推理
                outputs = infer(self.model, llm_inputs, self.sampling_params)
                
                # 整合结果
                for raw_input, output in zip(raw_inputs, outputs):
                    answer = output.outputs[0].text
                    result = {
                        "pid": raw_input["pid"],
                        "query": raw_input["query"],
                        "video_info": raw_input.get("video_info", ""),
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    pending_results.append(result)
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {str(e)}")
                traceback.print_exc()
            
            # 定期保存结果
            if len(pending_results) >= self.config.save_interval:
                total_count = self.save_results(pending_results)
                pending_results = []
                print(f"已处理 {batch_idx+1}/{total_batches} 批，当前共 {total_count} 条结果")
            
            # 定期清理缓存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # 保存剩余结果
        if pending_results:
            total_count = self.save_results(pending_results)
            print(f"处理完成，最终共 {total_count} 条结果")
        
        # 计算总耗时
        end_time = time.time()
        total_time = end_time - start_time
        print(f"处理完成，总耗时: {total_time:.2f}秒")

def main():
    parser = argparse.ArgumentParser(description="视频问答推理")
    parser.add_argument("config_file", help="配置文件路径")
    args = parser.parse_args()
    
    # 验证配置文件存在
    if not os.path.exists(args.config_file):
        print(f"错误：配置文件 '{args.config_file}' 不存在")
        exit(1)
    
    cfg = OmegaConf.load(args.config_file)
    print("配置:", cfg)
    
    # 设置CUDA设备可见性（如果有指定）
    if hasattr(cfg, 'gpu_id') and cfg.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    
    torch.multiprocessing.set_start_method("spawn")
    worker = VideoQAInference(cfg)
    worker.run()

if __name__ == "__main__":
    main()

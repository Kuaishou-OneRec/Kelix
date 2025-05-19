import json
import pandas as pd
import os

def convert_json_to_parquet(json_file_list):
    """
    将JSON文件列表中的每个文件转换为同名Parquet文件
    :param json_file_list: JSON文件路径列表
    """
    for json_path in json_file_list:
        try:
            # 读取JSON数据
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 生成输出路径
            parquet_path = os.path.splitext(json_path)[0] + '.parquet'
            
            # 写入Parquet文件
            df.to_parquet(parquet_path, index=False)
            
            print(f"转换成功: {json_path} -> {parquet_path}")
        
        except json.JSONDecodeError as e:
            print(f"JSON解析错误 ({json_path}): {str(e)}")
        except FileNotFoundError:
            print(f"文件不存在: {json_path}")
        except Exception as e:
            print(f"处理文件时发生意外错误 ({json_path}): {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 替换为实际的JSON文件路径列表
    json_files = [
        "/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MME/MME.json",
        "/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MMTBench/mmt_bench_485_hetu_format.json",
        "/mmu_mllm_hdd/shiyaya/dataset/mm_reasoning/benchmark/MMStar/YuanQi/mmstar.json",
        "/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/MathVista/mathvista.json",
        "/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/flickr30k/flickr30k_karpathy_test.json",
        "/llm_reco_ssd/luoxinchen/RecoVLM/Benchmark/dataset/Benchmark_v21/Benchmark_v21.json",
        "/llm_reco_ssd/luoxinchen/dataset/infoVQA/human_download/infographicsvqa_qas/reconstruct_val.json"
    ]
    convert_json_to_parquet(json_files)
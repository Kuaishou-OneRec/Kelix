import os
import uuid
import pandas as pd
from pathlib import Path

def generate_unique_uuid():
    """生成唯一的UUID字符串"""
    return str(uuid.uuid4())

def process_parquet_file(input_path, output_dir, num_rows=1000, num_files=100):
    """
    处理parquet文件，复制第一行并生成唯一UUID，保存为多个文件
    
    Args:
        input_path: 输入parquet文件路径
        output_dir: 输出目录路径
        num_rows: 每个文件要生成的行数
        num_files: 要生成的文件数量
    """
    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 读取parquet文件
        df = pd.read_parquet(input_path)
        print(f"成功读取文件: {input_path}")
        print(f"原文件共有 {len(df)} 行数据")
        
        # 获取第一行数据
        first_row = df.iloc[0:1].copy()  # 保持DataFrame格式而不是Series
        print(f"成功提取第一行数据，包含字段: {list(first_row.columns)}")
        
        # 检查是否包含uuid字段
        if 'uuid' not in first_row.columns:
            raise ValueError("parquet文件中未找到'uuid'字段！")
        
        # 生成100份文件
        for file_idx in range(num_files):
            # 准备存储1000行数据的列表
            new_rows = []
            
            # 生成1000行数据，每行使用不同的uuid
            for _ in range(num_rows):
                # 复制第一行数据
                row_copy = first_row.copy()
                # 生成新的唯一uuid
                row_copy['uuid'] = generate_unique_uuid()
                new_rows.append(row_copy)
            
            # 合并所有行并保存
            new_df = pd.concat(new_rows, ignore_index=True)
            output_file = os.path.join(output_dir, f"generated_file_{file_idx:03d}.parquet")
            new_df.to_parquet(output_file, index=False)
            
            if (file_idx + 1) % 10 == 0:
                print(f"已生成 {file_idx + 1}/{num_files} 个文件")
        
        print(f"\n完成！共生成 {num_files} 个文件，每个文件包含 {num_rows} 行数据")
        print(f"文件保存路径: {output_dir}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

# 主程序
if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = '/llm_reco/lingzhixin/recovlm_data/datasets/Gen_qwen_image_position/0.0.0/part/rank0-0.parquet'
    OUTPUT_DIR = '/llm_reco/lingzhixin/recovlm_data/datasets/Gen_qwen_image_position/0.0.0/generated_repeated_files'  # 你可以修改这个输出目录
    
    # 执行处理
    process_parquet_file(
        input_path=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        num_rows=1000,  # 每个文件1000行
        num_files=100   # 生成100个文件
    )
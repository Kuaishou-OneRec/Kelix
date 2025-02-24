import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Tuple
import os

class ParquetBalancer:
    def read_parquet(self, input_path: str) -> pd.DataFrame:
        """
        读取本地 parquet 文件
        Args:
            input_path: parquet 文件路径
        Returns:
            pandas DataFrame
        """
        try:
            # 直接使用 pandas 读取 parquet 文件
            df = pd.read_parquet(input_path)
            return df
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            raise

    def get_balanced_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        平衡数据集中的满意/不满意比例
        Args:
            df: 输入的 DataFrame
        Returns:
            平衡后的 DataFrame 和统计信息
        """
        # 获取原始数据统计
        original_counts = df['wenjuan_type'].value_counts().to_dict()
        
        # 找出数量最少的类别的数量
        min_count = min(original_counts.values())
        
        # 分别获取满意和不满意的数据
        satisfied = df[df['wenjuan_type'] == '满意']
        unsatisfied = df[df['wenjuan_type'] == '不满意']
        
        # 根据数量决定哪个需要降采样
        if len(satisfied) > len(unsatisfied):
            satisfied = satisfied.sample(n=min_count, random_state=42)
        else:
            unsatisfied = unsatisfied.sample(n=min_count, random_state=42)
        
        # 合并数据
        balanced_df = pd.concat([satisfied, unsatisfied])
        # 随机打乱顺序
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 获取处理后的统计信息
        final_counts = balanced_df['wenjuan_type'].value_counts().to_dict()
        
        return balanced_df, {
            'original_counts': original_counts,
            'final_counts': final_counts
        }

    def write_parquet(self, df: pd.DataFrame, output_path: str) -> None:
        """
        将处理后的数据写入本地
        Args:
            df: 要写入的 DataFrame
            output_path: 输出路径
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 直接使用 pandas 写入 parquet 文件
            df.to_parquet(output_path)
        except Exception as e:
            print(f"写入文件失败: {str(e)}")
            raise

    def process_file(self, input_path: str, output_path: str) -> None:
        """
        处理主函数
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
        """
        try:
            # 读取数据
            print(f"正在读取文件: {input_path}")
            df = self.read_parquet(input_path)
            print(f"读取文件成功, 数据量: {len(df)}")
            print(f"数据列: {df.columns}")
            
            # 平衡数据
            print("正在平衡数据...")
            balanced_df, stats = self.get_balanced_data(df)
            
            # 输出统计信息
            print("\n原始数据统计:")
            for type_name, count in stats['original_counts'].items():
                print(f"{type_name}: {count}")
            
            print("\n处理后数据统计:")
            for type_name, count in stats['final_counts'].items():
                print(f"{type_name}: {count}")
            
            # 写入数据
            print(f"\n正在写入平衡后的数据到: {output_path}")
            self.write_parquet(balanced_df, output_path)
            
            print("处理完成!")
            
        except Exception as e:
            print(f"处理过程中发生错误: {str(e)}")
            raise

def main():
    # 配置参数
    INPUT_PATH = "/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_11w_cot_v2/photo_0210_11w_sft_data-test.parquet"  # 替换为你的输入文件路径
    OUTPUT_PATH = "/llm_reco_ssd/huqigen/dataset/wenjuan_sft/photo_0210_11w_cot_v2/photo_0210_11w_sft_data-test_balanced.parquet"  # 替换为你的输出文件路径
    
    # 创建处理器实例并执行
    balancer = ParquetBalancer()
    balancer.process_file(INPUT_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    main()
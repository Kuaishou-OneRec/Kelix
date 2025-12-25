import pandas as pd
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

source_parquets = [
    "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/GenUno1M/0.0.2/rank0-0.parquet",
    "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zangdunju/datasets/Gen_text_image/2.0.1/rank0_0/rank0-0.parquet",
    "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/Gen_BLIP3o-60k/2.0.0/rank0_0/rank0-0.parquet",
    "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/zhouyang12/datasets/Gen_midjourney-niji-1m-llavanext/1.0.0/rank0_0/rank0-0.parquet",
    "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/moe_xtr_0812/base/models/2508/liziming/datasets/Gen_qwen_image_mix/0.0.0/rank0_0/rank0-0.parquet",
    "viewfs://hadoop-lt-cluster/home/reco_wl/mpi/moe_xtr_0812/base/models/2508/liziming/datasets/Gen_qwen_image_position/0.0.0/rank0_0/rank0-0.parquet"
]

target_parquet = "/mmu_mllm_hdd_2/lingzhixin/recovlm_data/muse_v2/vis/vis_data1225.parquet"

def main():
    """
    主函数：读取所有源parquet文件的前2行，拼接后写入目标parquet文件
    """
    logger.info("开始处理parquet文件聚合任务")
    
    # 创建空列表存储所有数据帧
    all_data = []
    
    # 遍历所有源文件
    for idx, source_file in enumerate(source_parquets):
        logger.info(f"正在处理第 {idx + 1}/{len(source_parquets)} 个文件: {source_file}")
        
        try:
            # 读取文件的前2行
            df = pd.read_parquet(source_file, nrows=2)
            logger.info(f"成功读取 {len(df)} 行数据")
            
            # 检查是否有数据
            if not df.empty:
                # 添加源文件标识列（可选）
                df['source_file'] = source_file
                # 添加到列表
                all_data.append(df)
            else:
                logger.warning(f"文件 {source_file} 为空")
                
        except Exception as e:
            logger.error(f"处理文件 {source_file} 时出错: {str(e)}")
            continue
    
    # 检查是否有数据
    if not all_data:
        logger.error("没有成功读取到任何数据，任务失败")
        return
    
    # 拼接所有数据
    logger.info(f"开始拼接所有数据，共 {len(all_data)} 个数据帧")
    concatenated_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"拼接完成，总数据行数: {len(concatenated_df)}")
    
    # 创建目标目录（如果不存在）
    target_dir = os.path.dirname(target_parquet)
    if target_dir and not os.path.exists(target_dir):
        logger.info(f"创建目标目录: {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
    
    # 写入目标文件
    logger.info(f"正在写入目标文件: {target_parquet}")
    concatenated_df.to_parquet(target_parquet, index=False)
    
    logger.info("任务完成！")

if __name__ == "__main__":
    main()

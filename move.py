import os
import shutil

def copy_py_files(src_dir: str, dst_dir: str) -> None:
    """
    递归复制源文件夹及其子目录中所有的 .py 文件到目标文件夹，保留目录结构。
    
    参数:
        src_dir: 源文件夹路径（绝对路径或相对路径均可）
        dst_dir: 目标文件夹路径（若不存在会自动创建）
    """
    # 遍历源目录及其所有子目录
    for root, _, files in os.walk(src_dir):
        # 计算当前目录相对于源目录的相对路径（用于保留目录结构）
        relative_path = os.path.relpath(root, src_dir)
        # 构建目标目录中的对应子目录
        dst_subdir = os.path.join(dst_dir, relative_path)
        # 确保目标子目录存在（不存在则创建，已存在则跳过）
        os.makedirs(dst_subdir, exist_ok=True)
        
        # 遍历当前目录下的所有文件，筛选 .py 后缀
        for file in files:
            if file.endswith(".py"):
                src_file = os.path.join(root, file)  # 源文件完整路径
                dst_file = os.path.join(dst_subdir, file)  # 目标文件完整路径
                # 复制文件（保留修改时间、权限等元数据）
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} → {dst_file}")




# vimdiff /llm_reco_ssd/caojiangxia/vllm/recovlm/recovlm/data/datasets.py /llm_reco/lingzhixin/recovlm_cjx_0707/recovlm/recovlm/data/datasets.py 
copy_py_files("/llm_reco_ssd/caojiangxia/vllm/recovlm", "/llm_reco/lingzhixin/recovlm_cjx_0707/recovlm")








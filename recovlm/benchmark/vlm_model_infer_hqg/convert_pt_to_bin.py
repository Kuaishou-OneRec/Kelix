import torch
import argparse

def convert_model(model_path, bin_path):
    # 加载模型
    model = torch.load(model_path)
    
    # 保存为 .bin 文件
    torch.save(model["module"], bin_path)
    print(f"Model successfully converted from {model_path} to {bin_path}")

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Convert a PyTorch model from .pt to .bin format.")
    
    # 添加命令行参数
    parser.add_argument("model_path", type=str, help="Path to the input .pt model file")
    parser.add_argument("bin_path", type=str, help="Path to the output .bin file")

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数
    convert_model(args.model_path, args.bin_path)
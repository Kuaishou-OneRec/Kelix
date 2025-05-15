import platform
import subprocess
import os
import re

def get_gpu_model():
    """
    获取当前系统中NVIDIA显卡的型号信息
    
    返回:
    str: 显卡型号名称，如果无法检测则返回 "Unknown"
    """
    try:
        # 优先尝试使用nvidia-smi（最可靠的方法）
        if platform.system() in ["Linux", "Darwin"]:  # Linux/macOS
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        
        elif platform.system() == "Windows":  # Windows
            # 尝试使用nvidia-smi（如果在PATH中）
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                shell=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # 备选方案：使用Windows Management Instrumentation (WMI)
            try:
                import wmi
                c = wmi.WMI()
                gpus = c.Win32_VideoController()
                for gpu in gpus:
                    if "NVIDIA" in gpu.Name:
                        return gpu.Name
            except ImportError:
                pass
    
        # 备选方案：检查CUDA库（需要PyTorch或TensorFlow）
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            if tf.test.is_gpu_available():
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    return details.get('device_name', 'NVIDIA GPU')
        except ImportError:
            pass
    
        # 最后手段：检查系统环境变量或驱动文件
        if platform.system() == "Linux":
            # 检查驱动文件
            if os.path.exists("/proc/driver/nvidia/version"):
                with open("/proc/driver/nvidia/version", "r") as f:
                    first_line = f.readline().strip()
                    match = re.search(r"NVIDIA driver \S+ for (\S+)", first_line)
                    if match:
                        return match.group(1)
    
    except Exception as e:
        print(f"检测显卡型号时出错: {e}")
    
    return "Unknown"


def is_h800():
    gpu_model = get_gpu_model()
    return gpu_model.split('\n')[0].strip()=='NVIDIA H800'

# 使用示例
if __name__ == "__main__":
    gpu_model = get_gpu_model()
    print(f"当前检测到的显卡型号: [{gpu_model}]", gpu_model=='NVIDIA H800')

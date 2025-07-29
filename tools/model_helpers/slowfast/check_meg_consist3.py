import torch
import numpy as np

def compare_pth_tensors(pth_path1, pth_path2):
    """
    比较两个.pth文件中存储的tensor字典的差异，并在函数内部打印结果
    
    参数:
        pth_path1: 第一个.pth文件的路径
        pth_path2: 第二个.pth文件的路径
    """
    # 加载两个pth文件
    dict1 = torch.load(pth_path1)
    dict2 = torch.load(pth_path2)
    
    # 获取共同的key
    common_keys = set(dict1.keys()) & set(dict2.keys())
    results = {}
    
    print("="*80)
    print(f"正在比较两个文件: {pth_path1} 和 {pth_path2}")
    print("="*80)
    
    # 处理共同的key
    if common_keys:
        print(f"\n找到 {len(common_keys)} 个共同的键，开始比较...\n")
        
        for key in sorted(common_keys):  # 排序以便输出更有序
            tensor1 = dict1[key]
            tensor2 = dict2[key]
            
            # 确保是tensor类型
            if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
                print(f"⚠️ 警告: 键 '{key}' 不是tensor类型，已跳过")
                continue
            
            # 展平tensor
            flat1 = tensor1.flatten()
            flat2 = tensor2.flatten()
            
            # 计算需要保留的长度（取较短的那个）
            min_length = min(len(flat1), len(flat2))
            
            # 裁剪到相同长度
            flat1_trimmed = flat1[:min_length]
            flat2_trimmed = flat2[:min_length]
            
            # 取前90%的元素
            ninety_percent = int(0.9 * min_length)
            if ninety_percent == 0:
                print(f"⚠️ 警告: 键 '{key}' 裁剪后有效长度为0，已跳过")
                continue
                
            flat1_used = flat1_trimmed[:ninety_percent]
            flat2_used = flat2_trimmed[:ninety_percent]
            
            # 计算差异
            diff = flat1_used - flat2_used
            abs_diff = torch.abs(diff)
            
            # 打印统计信息
            print(f"{'#'*60}")
            print(f"键: {key}")
            print(f"{'#'*60}")
            print(f"  原始形状:")
            print(f"    tensor1: {tensor1.shape}")
            print(f"    tensor2: {tensor2.shape}")
            print(f"  展平后长度:")
            print(f"    tensor1: {len(flat1)}, tensor2: {len(flat2)}")
            print(f"    裁剪后长度: {min_length}")
            print(f"    用于统计的长度 (前90%): {ninety_percent}")
            print(f"  误差统计:")
            print(f"    平均绝对误差 (MAE): {torch.mean(abs_diff).item():.6f}")
            print(f"    最大绝对误差: {torch.max(abs_diff).item():.6f}")
            print(f"    最小绝对误差: {torch.min(abs_diff).item():.6f}")
            print(f"    中位数绝对误差: {torch.median(abs_diff).item():.6f}")
            print(f"    均方误差 (MSE): {torch.mean(diff **2).item():.6f}")
            print(f"    均方根误差 (RMSE): {torch.sqrt(torch.mean(diff** 2)).item():.6f}")
            print(f"    总绝对误差: {torch.sum(abs_diff).item():.6f}\n")
    else:
        print("\n⚠️ 警告: 两个文件没有共同的键")
    
    # 打印只在一个文件中存在的key
    only_in1 = set(dict1.keys()) - set(dict2.keys())
    only_in2 = set(dict2.keys()) - set(dict1.keys())
    
    if only_in1:
        print(f"{'#'*60}")
        print(f"只在第一个文件中存在的键 ({len(only_in1)} 个):")
        for key in sorted(only_in1):
            print(f"  - {key}")
    
    if only_in2:
        print(f"{'#'*60}")
        print(f"只在第二个文件中存在的键 ({len(only_in2)} 个):")
        for key in sorted(only_in2):
            print(f"  - {key}")
    
    print(f"\n{'='*80}")
    print("比较完成")
    print(f"{'='*80}")

# 使用示例
if __name__ == "__main__":
    # 比较两个文件（结果会直接打印）
    compare_pth_tensors(
        "/mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.1/Stage3_SlowFast/8b/slowfast_0723/compare_project_banoutput/hidden_states_recovlm.pth", 
        "/mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.1/Stage3_SlowFast/8b/slowfast_0723/compare_project_banoutput/hidden_states_meg.pth"
    )
    
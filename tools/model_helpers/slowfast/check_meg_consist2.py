import torch
import numpy as np

# 定义文件路径
path1 = "/mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.1/Stage3_SlowFast/8b/slowfast_0723/before_lllm/before_llm.pth"
path2 = "/mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.1/Stage3_SlowFast/8b/slowfast_0723/before_lllm/recovlm_before_llm.pth"

def load_embeds(path):
    """加载pth文件并提取inputs_embeds张量"""
    try:

        checkpoint = torch.load(path, map_location="cpu")
        if "inputs_embeds" not in checkpoint:
            res = checkpoint["input_embeds"].float()
        else: res = checkpoint["inputs_embeds"].float()  # 转为float32统一精度
        if res.ndim == 3:
            res = res[:,0]
        res = res[:10]
        print(path)
        print(res)
        return res
    except Exception as e:
        print(f"加载文件 {path} 失败: {str(e)}")
        return None

# 加载两个嵌入张量
embeds1 = load_embeds(path1)
embeds2 = load_embeds(path2)

if embeds1 is None or embeds2 is None:
    print("无法继续比较，因为至少一个嵌入张量加载失败")
else:
    # 1. 基本信息比较
    print("="*60)
    print("【基本信息比较】")
    print(f"张量1形状: {embeds1.shape}")
    print(f"张量2形状: {embeds2.shape}")
    print(f"形状是否一致: {'是' if embeds1.shape == embeds2.shape else '否'}")
    print(f"张量1数据类型: {embeds1.dtype}")
    print(f"张量2数据类型: {embeds2.dtype}")
    print(f"张量1元素数量: {embeds1.numel()}")
    print(f"张量2元素数量: {embeds2.numel()}")

    # 2. 数值误差计算（仅在形状一致时计算）
    print("\n" + "="*60)
    print("【数值误差指标】")
    if embeds1.shape != embeds2.shape:
        print("警告: 张量形状不一致，无法计算数值误差")
    else:
        # 计算基础差异
        diff = embeds1 - embeds2
        abs_diff = torch.abs(diff)
        
        # 相对误差（避免除以零）
        with torch.no_grad():
            denom = torch.max(torch.abs(embeds1), torch.abs(embeds2))
            denom = torch.where(denom < 1e-12, 1e-12, denom)  # 防止除零
            rel_diff = abs_diff / denom
        
        # 输出误差指标
        print(f"最大绝对误差: {abs_diff.max().item()}")
        print(f"平均绝对误差: {abs_diff.mean().item()}")
        print(f"中位数绝对误差: {torch.median(abs_diff).item()}")
        print(f"L1范数 (总绝对误差): {abs_diff.sum().item()}")
        print(f"L2范数 (欧氏距离): {torch.norm(diff).item()}")
        print(f"均方误差 (MSE): {torch.mean(diff **2).item()}")
        print(f"均方根误差 (RMSE): {torch.sqrt(torch.mean(diff** 2)).item()}")
        print(f"最大相对误差: {rel_diff.max().item()}")
        print(f"平均相对误差: {rel_diff.mean().item()}")

    # 3. 统计特性比较
    print("\n" + "="*60)
    print("【统计特性比较】")
    # 展平张量用于统计分析
    flat1 = embeds1.flatten()
    flat2 = embeds2.flatten()
    
    # 基本统计量
    print(f"张量1均值: {flat1.mean().item():.6f}")
    print(f"张量2均值: {flat2.mean().item():.6f}")
    print(f"均值绝对差: {torch.abs(flat1.mean() - flat2.mean()).item()}")
    
    print(f"\n张量1标准差: {flat1.std().item():.6f}")
    print(f"张量2标准差: {flat2.std().item():.6f}")
    print(f"标准差绝对差: {torch.abs(flat1.std() - flat2.std()).item()}")
    
    print(f"\n张量1最大值: {flat1.max().item():.6f}")
    print(f"张量2最大值: {flat2.max().item():.6f}")
    print(f"最大值绝对差: {torch.abs(flat1.max() - flat2.max()).item()}")
    
    print(f"\n张量1最小值: {flat1.min().item():.6f}")
    print(f"张量2最小值: {flat2.min().item():.6f}")
    print(f"最小值绝对差: {torch.abs(flat1.min() - flat2.min()).item()}")
    
    # 分布相关性
    if embeds1.numel() > 0 and embeds2.numel() > 0:
        # 余弦相似度（方向一致性）
        cos_sim = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)).item()
        print(f"\n余弦相似度 (方向一致性): {cos_sim:.6f}")
        print(f"余弦距离 (1 - 余弦相似度): {1 - cos_sim}")
        
        # 皮尔逊相关系数（线性相关性）
        mean1 = flat1.mean()
        mean2 = flat2.mean()
        cov = ((flat1 - mean1) * (flat2 - mean2)).mean()
        corr = cov / (flat1.std() * flat2.std() + 1e-12)
        print(f"皮尔逊相关系数 (线性相关性): {corr.item():.6f}")

    print("\n" + "="*60)
    print("比较完成")
    print("="*60)


'''
============================================================
【基本信息比较】
张量1形状: torch.Size([3000, 4096])
张量2形状: torch.Size([3000, 4096])
形状是否一致: 是
张量1数据类型: torch.float32
张量2数据类型: torch.float32
张量1元素数量: 12288000
张量2元素数量: 12288000

============================================================
【数值误差指标】
最大绝对误差: 26.2265625
平均绝对误差: 0.9704400897026062
中位数绝对误差: 0.65234375
L1范数 (总绝对误差): 11924768.0
L2范数 (欧氏距离): 5123.50732421875
均方误差 (MSE): 2.137725830078125
均方根误差 (RMSE): 1.4620963335037231
最大相对误差: 2.0
平均相对误差: 0.7319422364234924

============================================================
【统计特性比较】
张量1均值: 0.018065
张量2均值: -0.011425
均值绝对差: 0.029489506036043167

张量1标准差: 1.361313
张量2标准差: 0.744344
标准差绝对差: 0.6169698238372803

张量1最大值: 25.500000
张量2最大值: 8.937500
最大值绝对差: 16.5625

张量1最小值: -18.375000
张量2最小值: -6.625000
最小值绝对差: 11.75

余弦相似度 (方向一致性): 0.133264
余弦距离 (1 - 余弦相似度): 0.8667359501123428
皮尔逊相关系数 (线性相关性): 0.133410

============================================================
比较完成
============================================================


只是文本
tensor([[ 3.1281e-04,  1.3428e-02, -1.4832e-02,  ..., -9.3384e-03,
         -6.1951e-03,  4.2114e-03],
        [ 2.1851e-02, -4.0039e-02,  1.2207e-02,  ..., -2.9907e-03,
         -1.3794e-02, -3.6865e-02],
        [-5.0354e-04,  4.5471e-03,  5.4016e-03,  ..., -1.0925e-02,
         -8.9722e-03,  5.1117e-04],
        ...,
        [ 9.3384e-03, -9.5215e-03,  3.3691e-02,  ..., -2.2278e-03,
          6.1523e-02,  4.2725e-03],
        [-2.0752e-03, -2.5177e-03,  6.8359e-03,  ..., -3.1948e-05,
         -1.2207e-02,  2.2278e-03],
        [ 3.1738e-03,  6.8665e-03, -1.3306e-02,  ..., -1.2085e-02,
          2.8419e-04,  1.6846e-02]])
============================================================
【基本信息比较】
张量1形状: torch.Size([10, 4096])
张量2形状: torch.Size([10, 4096])
形状是否一致: 是
张量1数据类型: torch.float32
张量2数据类型: torch.float32
张量1元素数量: 40960
张量2元素数量: 40960

============================================================
【数值误差指标】
最大绝对误差: 0.0001220703125
平均绝对误差: 2.0670277081080712e-05
中位数绝对误差: 0.0
L1范数 (总绝对误差): 0.8466545343399048
L2范数 (欧氏距离): 0.00741053931415081
均方误差 (MSE): 1.3407250865071774e-09
均方根误差 (RMSE): 3.6615914723370224e-05
最大相对误差: 1.9583333730697632
平均相对误差: 0.010421638377010822

============================================================
【统计特性比较】
张量1均值: 0.000009
张量2均值: 0.000009
均值绝对差: 3.9956830732990056e-08

张量1标准差: 0.024401
张量2标准差: 0.024402
标准差绝对差: 3.296881914138794e-07

张量1最大值: 0.535156
张量2最大值: 0.535156
最大值绝对差: 0.0

张量1最小值: -0.621094
张量2最小值: -0.621094
最小值绝对差: 0.0

余弦相似度 (方向一致性): 0.999999
余弦距离 (1 - 余弦相似度): 6.556510925292969e-07
皮尔逊相关系数 (线性相关性): 0.999975

============================================================
比较完成
============================================================
'''
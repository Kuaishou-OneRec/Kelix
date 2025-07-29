import os
import pickle
import torch
import torch.nn.functional as F




pref = "/mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.1/Stage3_SlowFast/8b/slowfast_0723/compare_project_banoutput"
# pref = "/mmu_mllm_hdd_2/lingzhixin/output1/Keye/0.9.1/Stage3_SlowFast/8b/slowfast_0723/compare_project_banoutput_fix_scatter/"
path = [
    f"{pref}/meg_entropy.pth",
    f"{pref}/recovlm_entropy.pth"
]

def load(p):
    x = torch.load(p)
    
    # 3272 v.s. 4000
    label = x["labels151936"]
    if len(label.shape) != 2: label = label[None]
    label = label[:,:3000]
    logits = x["logits151936"]
    if len(logits.shape) != 3 : logits = logits[None]
    logits = logits[:,:3000, :151936]
    print(p, logits.shape, label.shape)
    print(logits[:,:,:100])
    return label, logits

labels1, logits1 = load(path[0])
labels2, logits2 = load(path[1])

# 检查张量形状是否匹配的辅助函数
def check_shape_match(tensor1, tensor2, name):
    if tensor1.shape != tensor2.shape:
        print(f"{name}形状不匹配: {tensor1.shape} vs {tensor2.shape}")
        return False
    return True

# 1. 比较labels（忽略-100的位置）
print("=== 比较labels ===")
if not check_shape_match(labels1, labels2, "labels"):
    pass
else:
    # 创建忽略-100的掩码
    ignore_mask = (labels1 != -100) & (labels2 != -100)
    valid_count = torch.sum(ignore_mask).item()
    
    if valid_count == 0:
        print("没有有效的label用于比较（全部为-100）")
    else:
        # 只比较非-100的位置
        valid_labels1 = labels1[ignore_mask]
        valid_labels2 = labels2[ignore_mask]
        assert torch.all(valid_labels1 == valid_labels2)

        # 检查是否所有有效元素都相同
        all_equal = torch.all(torch.eq(valid_labels1, valid_labels2)).item()
        if all_equal:
            print(f"在所有非-100的位置，labels完全相同（共{valid_count}个有效位置）")
        else:
            # 统计差异
            diff_count = torch.sum(torch.ne(valid_labels1, valid_labels2)).item()
            diff_percent = (diff_count / valid_count) * 100
            print(f"labels存在差异: {diff_count}/{valid_count} 个有效元素不同 ({diff_percent:.2f}%)")
            
            # 显示第一个差异的位置和值
            if diff_count > 0:
                # 找到原始张量中的第一个差异位置
                diff_mask = (labels1 != labels2) & ignore_mask
                first_diff = torch.nonzero(diff_mask).squeeze()[0].tolist()
                print(f"第一个差异位置: {first_diff}, " 
                      f"值分别为: {labels1[0, first_diff]} 和 {labels2[0, first_diff]}")

# 2. 比较logits（增加详细的差异指标）
print("\n=== 比较logits ===")
if not check_shape_match(logits1, logits2, "logits"):
    pass
else:
    # 计算差异值
    diff_values = torch.abs(logits1 - logits2)
    total_elements = logits1.numel()
    
    # 检查是否所有元素都相同（考虑浮点数精度问题）
    atol = 1e-6  # 绝对误差容限
    all_close = torch.allclose(logits1, logits2, atol=atol)
    diff_mask = ~torch.isclose(logits1, logits2, atol=atol)
    diff_count = torch.sum(diff_mask).item()
    diff_percent = (diff_count / total_elements) * 100
    
    if all_close:
        print(f"logits在允许的误差范围内完全相同（容限: {atol}）")
    else:
        print(f"logits存在差异: {diff_count}/{total_elements} 个元素不同 ({diff_percent:.2f}%)")
    
    # 计算整体差异统计指标（包括所有元素）
    print("\n整体差异统计指标（包括所有元素）:")
    print(f"  MAE (平均绝对误差): {torch.mean(diff_values.abs()).item():.8f}")
    print(f"  MRE (平均绝对误差): {torch.mean(diff_values.abs()).item() / torch.mean(logits2.abs()).item():.8f}")
    print(f"  中位数绝对误差: {torch.median(diff_values).item():.8f}")
    print(f"  标准差: {torch.std(diff_values).item():.8f}")
    print(f"  最大值: {torch.max(diff_values).item():.8f}")
    print(f"  最小值: {torch.min(diff_values).item():.8f}")
    print(f"  总和: {torch.sum(diff_values).item():.8f}")
    
    # 计算仅差异元素的统计指标（只考虑差异超过阈值的元素）
    if diff_count > 0:
        diff_only = diff_values[diff_mask]
        print("\n仅差异元素的统计指标（超过误差容限的元素）:")
        print(f"  MAE (平均绝对误差): {torch.mean(diff_only.abs()).item():.8f}")
        print(f"  MRE (平均绝对误差): {torch.sum(diff_only.abs()).item() / torch.sum(logits2[diff_mask].abs()).item():.8f}")
        print(f"  中位数绝对误差: {torch.median(diff_only).item():.8f}")
        print(f"  标准差: {torch.std(diff_only).item():.8f}")
        print(f"  最大值: {torch.max(diff_only).item():.8f}")
        print(f"  最小值: {torch.min(diff_only).item():.8f}")
        print(f"  总和: {torch.sum(diff_only).item():.8f}")
        

# 3. 比较交叉熵损失（忽略label=-100的位置）
print("\n=== 比较交叉熵损失 ===")
try:
    # 确保labels是长整型（交叉熵要求的格式）
    labels1_long = labels1.long().squeeze(0)  # 移除1x4000中的1维度
    labels2_long = labels2.long().squeeze(0)
    
    # 移除logits中的1维度 (1x4000x151936 -> 4000x151936)
    logits1_squeezed = logits1.squeeze(0)
    logits2_squeezed = logits2.squeeze(0)
    
    print(logits1_squeezed[labels1_long != -100].shape)
    print(labels1_long[labels1_long != -100].shape)


    # 计算交叉熵损失，忽略label=-100的位置
    loss1 = F.cross_entropy(logits1_squeezed.float(), labels1_long, ignore_index=-100)
    loss2 = F.cross_entropy(logits2_squeezed.float(), labels2_long, ignore_index=-100)
    print(loss1)

    # 统计有效标签数量（非-100的标签）
    valid_labels_count1 = torch.sum(labels1_long != -100).item()
    valid_labels_count2 = torch.sum(labels2_long != -100).item()
    
    print(f"第一个文件夹的有效标签数量: {valid_labels_count1}")
    print(f"第二个文件夹的有效标签数量: {valid_labels_count2}")
    print(f"第一个文件夹的交叉熵损失: {loss1.item():.6f}")
    print(f"第二个文件夹的交叉熵损失: {loss2.item():.6f}")
    print(f"损失差异: {abs(loss1 - loss2).item():.6f}")
    
    if torch.isclose(loss1, loss2, atol=1e-6):
        print("交叉熵损失在允许的误差范围内相同")
    else:
        print("交叉熵损失存在显著差异")
except Exception as e:
    print(f"计算交叉熵损失时出错: {e}")




'''
import torch

# 假设 logits151936 和 labels151936 已定义
# logits151936: 形状 [4000, 155136]，未经过softmax的原始输出
# labels151936: 形状 [4000]，每个元素是类别索引（0 ~ 155135）

# 步骤1：对logits做数值稳定处理（减去每行最大值，避免exp溢出）
# 计算每行的最大值（保持维度，方便广播）
max_logits = torch.max(logits151936, dim=1, keepdim=True).values  # 形状 [4000, 1]
logits_stable = logits151936 - max_logits  # 形状 [4000, 155136]

# 步骤2：计算softmax概率
exp_logits = torch.exp(logits_stable)  # 形状 [4000, 155136]
sum_exp = torch.sum(exp_logits, dim=1, keepdim=True)  # 每行的指数和，形状 [4000, 1]
softmax_probs = exp_logits / sum_exp  # 形状 [4000, 155136]

# 步骤3：提取每个样本对应标签的概率
# 生成样本索引（0 ~ 3999）
batch_indices = torch.arange(logits151936.size(0), device=logits151936.device)  # 形状 [4000]
# 高级索引：选取每个样本中标签对应的概率
selected_probs = softmax_probs[batch_indices, labels151936]  # 形状 [4000]

# 步骤4：计算负对数（加微小值避免log(0)）
log_probs = torch.log(selected_probs + 1e-10)  # 形状 [4000]
negative_log_probs = -log_probs  # 形状 [4000]

# 步骤5：求平均得到交叉熵损失
cross_entropy_loss = torch.mean(negative_log_probs)  # 标量

print("手动计算的交叉熵损失值:", cross_entropy_loss.item())

'''

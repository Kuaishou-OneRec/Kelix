import torch

def process_pos_ids(pos_ids, input_ids):
    """
    处理positional id，基于外部提供的图像标记区分图片/非图片token
    
    规则：
    - 非图片token（is_image_token=False）：t/h/w均为0
    - 图片token（is_image_token=True）：
      - t：每张图片内部从1开始递增
      - h：每张图片内部从1开始递增（行坐标）
      - w：每张图片内部从1开始递增（列坐标）
    """
    # 提取t/h/w维度（形状：[N]）
    t = pos_ids[0, 0]
    h = pos_ids[1, 0]
    w = pos_ids[2, 0]
    device = t.device
    N = t.numel()  # 总token数

    is_image_token = (input_ids == fast_vid_pad) | (input_ids == vid_pad) | (input_ids == image_pad)
    is_image_token = is_image_token[0]

    # 转换图像标记为张量
    is_image = torch.tensor(is_image_token, dtype=torch.bool, device=device)
    
    # 初始化结果张量为长整型（解决类型不匹配问题）
    new_t = torch.zeros(N, device=device, dtype=torch.long)
    new_h = torch.zeros(N, device=device, dtype=torch.long)
    new_w = torch.zeros(N, device=device, dtype=torch.long)
    
    # 处理图像token
    if is_image.any():
        # 获取图像token索引
        img_idx = torch.where(is_image)[0]  # 图像token位置
        
        # 计算图像分组（连续索引为同一图像）
        if len(img_idx) > 1:
            group_flags = (img_idx[1:] - img_idx[:-1] > 1)  # 新图像标记
            groups = torch.cumsum(torch.cat([torch.tensor([1], device=device), group_flags]), 0) - 1
        else:
            groups = torch.tensor([0], device=device) if len(img_idx) == 1 else torch.tensor([], device=device)
        
        # 按图像分组处理
        for g in torch.unique(groups):
            mask = (groups == g)
            indices = img_idx[mask]  # 当前图像的所有token索引
            k = len(indices)  # 当前图像的token数量
            
            # 处理t维度：1~k递增
            new_t[indices] = torch.arange(1, k+1, device=device)
            
            # 处理h维度：组内最小值为基准
            group_h = h[indices]
            new_h[indices] = group_h - group_h.min() + 1
            
            # 处理w维度：组内最小值为基准
            group_w = w[indices]
            new_w[indices] = group_w - group_w.min() + 1
    
    # 重组为原始形状[3,1,N]
    return torch.stack([
        new_t.unsqueeze(0),
        new_h.unsqueeze(0),
        new_w.unsqueeze(0)
    ], dim=0)

# 测试用例
if __name__ == "__main__":
    

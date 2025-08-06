import torch
def process_pos_ids(pos_ids):
    """
    扩展版positional id处理函数（支持t/h/w全维度处理）
    
    功能：
        针对Qwen2 VL模型的positional id进行转换，规则如下：
        1. 非图片token（文本）：
            - t维度 → 0
            - h/w维度 → 0
        2. 图片token：
            - t维度 → 每张图片内部从1开始递增（独立计数）
            - h维度 → 每张图片内部从1开始递增（行坐标）
            - w维度 → 每张图片内部从1开始递增（列坐标）
    
    参数：
        pos_ids: torch.Tensor，形状为[3, 1, N]，其中：
            - 第0维：t维度（时序ID）
            - 第1维：h维度（高度坐标）
            - 第2维：w维度（宽度坐标）
            - N为总token数量
    
    返回：
        torch.Tensor，形状与输入一致[3, 1, N]，所有维度按规则转换
    """
    # -------------------------- 1. 维度处理与初始化 --------------------------
    # 提取t/h/w维度（挤压中间维度，从[3,1,N]→3个[N]张量）
    t = pos_ids[0, 0]  # 形状: [N]
    h = pos_ids[1, 0]  # 形状: [N]
    w = pos_ids[2, 0]  # 形状: [N]
    
    device = h.device  # 设备兼容性处理
    n = t.numel()      # 总token数量
    
    # -------------------------- 2. 识别图像/文本token --------------------------
    # 2.1 识别图像token（h值重复出现≥2次）
    unique_h, counts = torch.unique(h, return_counts=True)
    image_h_values = unique_h[counts >= 2]
    is_image_token = torch.isin(h, image_h_values)  # 形状: [N]
    
    # 2.2 识别文本token（非图像token且t=h=w）
    is_text_token = ~is_image_token & (t == h) & (h == w)  # 形状: [N]
    
    # -------------------------- 3. 初始化转换后的张量 --------------------------
    new_t = torch.zeros_like(t)  # 非图片token默认0
    new_h = torch.zeros_like(h)  # 非图片token默认0
    new_w = torch.zeros_like(w)  # 非图片token默认0
    
    # -------------------------- 4. 处理图像token（t/h/w同步转换） --------------------------
    if is_image_token.any():
        # 4.1 获取图像token索引并分组（每张图片为一个连续片段）
        img_indices = torch.where(is_image_token)[0]  # 所有图像token的位置
        
        # 计算分组标记（连续索引为同一图像）
        if img_indices.numel() > 1:
            index_diff = img_indices[1:] - img_indices[:-1]
            new_group_flags = (index_diff > 1)  # 非连续处为新图像
        else:
            new_group_flags = torch.tensor([], device=device)
        
        # 生成每个token的分组ID
        groups = torch.cumsum(
            torch.cat([torch.tensor([1], device=device), new_group_flags]), 
            dim=0
        ) - 1  # 形状: [M]，M为图像token总数
        
        # 4.2 逐个图像处理（t/h/w独立从1开始递增）
        for group_id in torch.unique(groups):
            # 当前图像的所有token索引
            group_mask = (groups == group_id)
            seg_indices = img_indices[group_mask]  # 形状: [K]，K为当前图像token数
            
            # -------------------- 处理t维度：从1开始递增 --------------------
            new_t[seg_indices] = torch.arange(1, len(seg_indices)+1, device=device)
            
            # -------------------- 处理h维度：行坐标从1开始 --------------------
            seg_h = h[seg_indices]
            min_h = seg_h.min()
            new_h[seg_indices] = seg_h - min_h + 1
            
            # -------------------- 处理w维度：列坐标从1开始 --------------------
            seg_w = w[seg_indices]
            min_w = seg_w.min()
            new_w[seg_indices] = seg_w - min_w + 1
    
    # -------------------------- 5. 重组为原始形状 --------------------------
    processed = torch.stack([
        new_t.unsqueeze(0),  # 处理后的t维度
        new_h.unsqueeze(0),  # 处理后的h维度
        new_w.unsqueeze(0)   # 处理后的w维度
    ], dim=0)  # 最终形状[3,1,N]
    
    return processed



# 测试函数（验证功能正确性）
if __name__ == "__main__":
    # 构造测试用例：[t1, t2, t3, image(4×2), t4, t5, image(3×2), t6]
    pos_ids = torch.tensor([
        [  # t维度（时序ID，连续递增）
            [0, 1, 2,          # t1, t2, t3
             3,3,3,3,3,3,3,3, # 第一个图像(4×2，8个patch)
             11,12,            # t4, t5
             13,13,13,13,13,13,# 第二个图像(3×2，6个patch)
             19]               # t6
        ],
        [  # h维度（图像h = 前序最大t+1 + 行索引）
            [0, 1, 2,          # t1-t3：h=t
             3,3,3,3,4,4,4,4, # 第一个图像：行0→3，行1→4（前序最大t=2）
             11,12,            # t4-t5：h=t
             13,13,13,14,14,14,# 第二个图像：行0→13，行1→14（前序最大t=12）
             19]               # t6：h=t
        ],
        [  # w维度（图像w = 前序最大t+1 + 列索引）
            [0, 1, 2,          # t1-t3：w=t
             3,4,5,6,3,4,5,6, # 第一个图像：列0-3→3-6（前序最大t=2）
             11,12,            # t4-t5：w=t
             13,14,15,13,14,15,# 第二个图像：列0-2→13-15（前序最大t=12）
             19]               # t6：w=t
        ]
    ])
    
    # 处理positional id
    processed = process_pos_ids(pos_ids)
    
    # 打印结果与预期对比
    print("=== 处理结果验证 ===")
    print("处理后t维度:", processed[0, 0].tolist())
    print("处理后h维度:", processed[1, 0].tolist())
    print("预期h维度:   [0,0,0,1,1,1,1,2,2,2,2,0,0,1,1,1,2,2,2,0]")
    print("\n处理后w维度:", processed[2, 0].tolist())
    print("预期w维度:   [0,0,0,1,2,3,4,1,2,3,4,0,0,1,2,3,1,2,3,0]")

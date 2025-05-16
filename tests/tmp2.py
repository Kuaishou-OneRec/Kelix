import torch

class TestRotaryEmbedding:
    def __init__(self, spatial_merge_size=2):
        self.spatial_merge_size = spatial_merge_size
    
    def rot_pos_emb(self, grid_thw):
        print(f"\n输入 grid_thw 形状: {grid_thw.shape if isinstance(grid_thw, torch.Tensor) else type(grid_thw)}")
        print(f"输入 grid_thw 值: {grid_thw}")
        
        pos_ids = []
        for t, h, w in grid_thw:
            # 高度位置编码
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            print(f"\n初始 hpos_ids 形状: {hpos_ids.shape}")
            
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            print(f"重塑后 hpos_ids 形状: {hpos_ids.shape}")
            
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            print(f"置换后 hpos_ids 形状: {hpos_ids.shape}")
            
            hpos_ids = hpos_ids.flatten()
            print(f"展平后 hpos_ids 形状: {hpos_ids.shape}")
            
            # 宽度位置编码
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            print(f"\n初始 wpos_ids 形状: {wpos_ids.shape}")
            
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            print(f"重塑后 wpos_ids 形状: {wpos_ids.shape}")
            
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            print(f"置换后 wpos_ids 形状: {wpos_ids.shape}")
            
            wpos_ids = wpos_ids.flatten()
            print(f"展平后 wpos_ids 形状: {wpos_ids.shape}")
            
            # 合并并复制时间维度
            pos_ids_per_grid = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            print(f"时间复制后 pos_ids_per_grid 形状: {pos_ids_per_grid.shape}")
            pos_ids.append(pos_ids_per_grid)
        
        # 拼接所有网格的位置编码
        pos_ids = torch.cat(pos_ids, dim=0)
        print(f"\n最终 pos_ids 形状: {pos_ids.shape}")
        
        # 假设rotary_pos_emb_full函数返回一个预计算的位置嵌入表
        # 这里用随机张量模拟
        max_grid_size = grid_thw[:, 1:].max()
        print(f"max_grid_size: {max_grid_size}")
        
        # 模拟self.rotary_pos_emb函数的输出
        # 实际实现中，这应该是一个预计算的位置嵌入表
        rotary_pos_emb_full = torch.randn(max_grid_size, max_grid_size, 64)
        print(f"rotary_pos_emb_full 形状: {rotary_pos_emb_full.shape}")
        
        # 索引和展平
        rotary_pos_emb = rotary_pos_emb_full[pos_ids[:, 0], pos_ids[:, 1]].flatten(1)
        print(f"最终 rotary_pos_emb 形状: {rotary_pos_emb.shape}")
        
        return rotary_pos_emb

# 创建测试实例并运行
if __name__ == "__main__":
    # 测试用例：多个网格尺寸
    grid_thw = torch.tensor([
        [2, 4, 4],  # 时间=2, 高度=4, 宽度=4
        [1, 8, 8]   # 时间=1, 高度=8, 宽度=8
    ])
    
    # 假设spatial_merge_size能整除所有网格的高度和宽度
    test = TestRotaryEmbedding(spatial_merge_size=2)
    result = test.rot_pos_emb(grid_thw)
    print(f"\n最终输出形状: {result.shape}")
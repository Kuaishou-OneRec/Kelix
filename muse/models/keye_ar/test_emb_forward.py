import torch
import numpy as np

# 创建输入数据
input_data = [
    [872, 151681, -100, -100, -100, -100, -100, -100, -100],
    [198, 151681, -100, -100, -100, -100, -100, -100, -100],
    [151652, 151681, -100, -100, -100, -100, -100, -100, -100],
    [158989, 163361, 172176, 176543, 191772, 198720, 201995, 209754, 151681],
    [158723, 161428, 172475, 183267, 189182, 195380, 203676, 215606, 151681],
    [152707, 161925, 170851, 182049, 190989, 200537, 203905, 211502, 151681],
    [155872, 161428, 171248, 181246, 186400, 195380, 203676, 210683, 151681],
    [157266, 164215, 175621, 180185, 188326, 197081, 208022, 216938, 151681],
    [152940, 160320, 176261, 181962, 188078, 198887, 203676, 212428, 151681],
    [152707, 167798, 174112, 182049, 187728, 194661, 206012, 214532, 151681],
    [156614, 166166, 172568, 178561, 190379, 197759, 207621, 212810, 151681],
    [152707, 160713, 172989, 179118, 192749, 195380, 207734, 216938, 151681],
    [159674, 161952, 169345, 181686, 191525, 195874, 204839, 210086, 151681],
    [152707, 161925, 171383, 182049, 190989, 200363, 201154, 214532, 151681],
    [156614, 163115, 172568, 182049, 190379, 196183, 206012, 213994, 151681],
    [159418, 161971, 171135, 183504, 189079, 199763, 204210, 213800, 151681],
    [153068, 166166, 174479, 178561, 187447, 199763, 204210, 214334, 151681],
    [152707, 161925, 174112, 182049, 187728, 199126, 201204, 214532, 151681],
    [153068, 160713, 171520, 178561, 192513, 195874, 203905, 211502, 151681],
    [151653, 151681, -100, -100, -100, -100, -100, -100, -100],
    [3555, 151681, -100, -100, -100, -100, -100, -100, -100],
    [594, 151681, -100, -100, -100, -100, -100, -100, -100],
    [2629, 151681, -100, -100, -100, -100, -100, -100, -100],
    [315, 151681, -100, -100, -100, -100, -100, -100, -100],
    [279, 151681, -100, -100, -100, -100, -100, -100, -100],
    [1156, 151681, -100, -100, -100, -100, -100, -100, -100],
    [220, 151681, -100, -100, -100, -100, -100, -100, -100]
]

extended_tokens = torch.tensor(input_data)

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 创建共享的embedding层，确保两个函数使用完全相同的权重
class SharedEmbedding:
    def __init__(self, vocab_size=50000, embed_dim=4096):
        # 创建一个固定的embedding权重
        self.weight = torch.randn(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        
    def __call__(self, input_ids):
        # 实现标准的embedding查找
        # 对于-100，返回0向量
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        if len(input_ids.shape) == 2:
            # 标准2D输入: [batch, seq_len]
            embeddings = torch.zeros(batch_size, seq_len, self.embed_dim)
            for i in range(batch_size):
                for j in range(seq_len):
                    token_id = input_ids[i, j].item()
                    if token_id >= 0 and token_id < len(self.weight):
                        embeddings[i, j] = self.weight[token_id]
        elif len(input_ids.shape) == 3:
            # 3D输入: [batch, seq_len, group]
            batch_size, seq_len, group_size = input_ids.shape
            embeddings = torch.zeros(batch_size, seq_len, group_size, self.embed_dim)
            for i in range(batch_size):
                for j in range(seq_len):
                    for k in range(group_size):
                        token_id = input_ids[i, j, k].item()
                        if token_id >= 0 and token_id < len(self.weight):
                            embeddings[i, j, k] = self.weight[token_id]
        else:
            raise ValueError(f"Unsupported input shape: {input_ids.shape}")
            
        return embeddings
    
    def to(self, device):
        self.weight = self.weight.to(device)
        return self

# 创建共享的embedding实例
shared_embedding = SharedEmbedding(vocab_size=50000, embed_dim=4096)

# 创建共享的pre_embedding（虽然用不到，但为了完整性）
class SharedPreEmbedding:
    def __init__(self, pre_embedding_size=1000, embed_dim=256):
        self.weight = torch.randn(pre_embedding_size, embed_dim)
        self.embed_dim = embed_dim
        
    def __call__(self, input_ids):
        batch_size, seq_len, group_size = input_ids.shape
        embeddings = torch.zeros(batch_size, seq_len, group_size, self.embed_dim)
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(group_size):
                    token_id = input_ids[i, j, k].item()
                    if 0 <= token_id < len(self.weight):
                        embeddings[i, j, k] = self.weight[token_id]
        return embeddings

# 创建pre_embedding（虽然我们不会用到，因为pre_embedding_size为None）
shared_pre_embedding = SharedPreEmbedding(pre_embedding_size=1000, embed_dim=256)

# 创建模拟的pre_embedding_linear
class MockLinear:
    def __init__(self, in_features=256, out_features=4096):
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.randn(out_features)
        
    def __call__(self, x):
        # 简化实现：实际应该是线性变换
        batch_size, seq_len, group_size, embed_dim = x.shape
        output = torch.zeros(batch_size, seq_len, group_size, 4096)
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(group_size):
                    output[i, j, k] = torch.matmul(self.weight, x[i, j, k]) + self.bias
        return output

# 现在创建两个测试类，确保它们使用完全相同的组件
class TestClass1:
    def __init__(self, shared_embedding):
        # 配置参数
        self.vocab_size = 32000
        self.n_q_tokens = 8
        
        # 模拟config对象
        class MockConfig:
            class VisionConfig:
                def __init__(self):
                    self.n_q_tokens = 8
            def __init__(self):
                self.vision_config = self.VisionConfig()
                self.vocab_size = 32000
                
        self.config = MockConfig()
        
        # 模拟model对象
        class MockModel:
            def __init__(self, embedding):
                self.embed_tokens = embedding
                self.pre_embedding_size = None
                self.pre_embedding_tokens = 32000
                # pre_embedding和pre_embedding_linear虽然不会被用到，但为了完整性
                self.pre_embedding = lambda x: x  # 占位符
                self.pre_embedding_linear = MockLinear()
                
        self.model = MockModel(shared_embedding)
        
    def infer_id_embs(self, extended_tokens, group_size=None):
        """
        第一个函数的实现
        """
        if group_size is None:
            group_size = self.config.vision_config.n_q_tokens + 1
            
        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        batch_size, compressed_len, dim = extended_tokens.shape
        input_ids_reshaped = extended_tokens

        # 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.config.vocab_size)

        first_token[(first_token>=self.config.vocab_size) | (first_token<0)] = 0
        text_embeds = self.model.embed_tokens(first_token)
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))

        if self.model.pre_embedding_size is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input[(vis_emb_input >= self.model.pre_embedding_tokens) | (vis_emb_input<0)] = 0
            stage1_embeds = self.model.pre_embedding(vis_emb_input).detach()
            stage1_embeds = self.model.pre_embedding_linear(stage1_embeds)
            visual_embeds_final = stage1_embeds
        else:
            stage2_embeds = self.model.embed_tokens(safe_visual_indices)
            visual_embeds_final = stage2_embeds

        mask_final = is_visual_group.unsqueeze(-1).expand_as(text_embeds)

        text_embeds = text_embeds[:,:,None]
        if group_size > 1:
            text_embeds = text_embeds.repeat_interleave(group_size - 1, dim=2)

        token_inputs_embeds = torch.where(mask_final[:, :, None, :], visual_embeds_final, text_embeds)
        return token_inputs_embeds

class TestClass2:
    def __init__(self, shared_embedding):
        # 直接设置参数
        self.vocab_size = 32000
        self.n_q_tokens = 8
        self.embed_tokens = shared_embedding
        self.pre_embedding_size = None
        self.pre_embedding_tokens = 32000
        # 虽然不会被用到，但为了完整性
        self.pre_embedding = lambda x: x
        self.pre_embedding_linear = MockLinear()
        
    def infer_id_embs(self, extended_tokens, group_size=None):
        """
        第二个函数的实现
        """
        if group_size is None:
            group_size = self.n_q_tokens + 1

        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        input_ids_reshaped = extended_tokens

        # 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.vocab_size)

        first_token[(first_token>=self.vocab_size) | (first_token<0)] = 0
        text_embeds = self.embed_tokens(first_token)
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))

        if self.pre_embedding_size is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input[(vis_emb_input >= self.pre_embedding_tokens) | (vis_emb_input<0)] = 0
            stage1_embeds = self.pre_embedding(vis_emb_input).detach()
            stage1_embeds = self.pre_embedding_linear(stage1_embeds)
            visual_embeds_final = stage1_embeds
        else:
            stage2_embeds = self.embed_tokens(safe_visual_indices)
            visual_embeds_final = stage2_embeds

        mask_final = is_visual_group.unsqueeze(-1).expand_as(text_embeds)

        text_embeds = text_embeds[:,:,None]
        if group_size > 1:
            text_embeds = text_embeds.repeat_interleave(group_size - 1, dim=2)

        token_inputs_embeds = torch.where(mask_final[:, :, None, :], visual_embeds_final, text_embeds)
        return token_inputs_embeds

# 创建测试实例，使用相同的embedding
test1 = TestClass1(shared_embedding)
test2 = TestClass2(shared_embedding)

# 运行测试
print("=== 测试第一个函数 ===")
output1 = test1.infer_id_embs(extended_tokens, group_size=9)
print(f"输出1形状: {output1.shape}")

print("\n=== 测试第二个函数 ===")
output2 = test2.infer_id_embs(extended_tokens, group_size=9)
print(f"输出2形状: {output2.shape}")

# 检查形状是否相同
print(f"\n=== 形状比较 ===")
print(f"输出1形状: {output1.shape}")
print(f"输出2形状: {output2.shape}")
print(f"形状是否相同: {output1.shape == output2.shape}")

# 检查数值是否完全相同
print(f"\n=== 数值比较 ===")
# 由于两个函数都使用相同的embedding，并且逻辑相同，输出应该完全一样
if torch.allclose(output1, output2, rtol=1e-5, atol=1e-8):
    print("✓ 输出数值完全相同！")
else:
    print("✗ 输出数值不同")
    
    # 找出差异
    diff = torch.abs(output1 - output2)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"最大差异: {max_diff.item()}")
    print(f"平均差异: {mean_diff.item()}")
    
    # 找出差异大于阈值的位置
    threshold = 1e-5
    num_different = (diff > threshold).sum().item()
    total_elements = output1.numel()
    print(f"差异大于{threshold}的元素数量: {num_different}/{total_elements} ({100*num_different/total_elements:.2f}%)")
    
    # 打印前几个差异最大的位置
    if num_different > 0:
        print("\n前10个差异最大的位置:")
        flat_diff = diff.flatten()
        top_indices = torch.topk(flat_diff, min(10, num_different)).indices
        for idx in top_indices:
            idx_tuple = np.unravel_index(idx.item(), output1.shape)
            print(f"位置{idx_tuple}: 输出1={output1[idx_tuple].item():.6f}, 输出2={output2[idx_tuple].item():.6f}, 差异={diff[idx_tuple].item():.6f}")

# 检查中间变量
print(f"\n=== 中间变量检查 ===")
# 检查reshape后的结果
reshaped1 = extended_tokens.reshape([extended_tokens.shape[0], -1, 9])
reshaped2 = extended_tokens.reshape([extended_tokens.shape[0], -1, 9])
print(f"reshape结果是否相同: {torch.all(reshaped1 == reshaped2)}")

# 检查first_token
first_token1 = reshaped1[:, :, 0].clone()
first_token2 = reshaped2[:, :, 0].clone()
print(f"first_token是否相同: {torch.all(first_token1 == first_token2)}")

# 检查is_visual_group
is_visual_group1 = (first_token1 >= test1.config.vocab_size)
is_visual_group2 = (first_token2 >= test2.vocab_size)
print(f"is_visual_group是否相同: {torch.all(is_visual_group1 == is_visual_group2)}")

# 检查处理后的first_token
first_token1_processed = first_token1.clone()
first_token1_processed[(first_token1_processed >= test1.config.vocab_size) | (first_token1_processed < 0)] = 0
first_token2_processed = first_token2.clone()
first_token2_processed[(first_token2_processed >= test2.vocab_size) | (first_token2_processed < 0)] = 0
print(f"处理后的first_token是否相同: {torch.all(first_token1_processed == first_token2_processed)}")

# 检查text_embeds
text_embeds1 = test1.model.embed_tokens(first_token1_processed)
text_embeds2 = test2.embed_tokens(first_token2_processed)
print(f"text_embeds是否相同: {torch.allclose(text_embeds1, text_embeds2, rtol=1e-5, atol=1e-8)}")

# 检查raw_visual_indices
raw_visual_indices1 = reshaped1[:, :, :-1]
raw_visual_indices2 = reshaped2[:, :, :-1]
print(f"raw_visual_indices是否相同: {torch.all(raw_visual_indices1 == raw_visual_indices2)}")

# 检查safe_visual_indices
safe_visual_indices1 = torch.where(is_visual_group1.unsqueeze(-1).expand_as(raw_visual_indices1), 
                                   raw_visual_indices1, torch.zeros_like(raw_visual_indices1))
safe_visual_indices2 = torch.where(is_visual_group2.unsqueeze(-1).expand_as(raw_visual_indices2), 
                                   raw_visual_indices2, torch.zeros_like(raw_visual_indices2))
print(f"safe_visual_indices是否相同: {torch.all(safe_visual_indices1 == safe_visual_indices2)}")

print(f"\n=== 配置参数检查 ===")
print(f"test1.config.vision_config.n_q_tokens: {test1.config.vision_config.n_q_tokens}")
print(f"test2.n_q_tokens: {test2.n_q_tokens}")
print(f"是否相同: {test1.config.vision_config.n_q_tokens == test2.n_q_tokens}")

print(f"\ntest1.config.vocab_size: {test1.config.vocab_size}")
print(f"test2.vocab_size: {test2.vocab_size}")
print(f"是否相同: {test1.config.vocab_size == test2.vocab_size}")

print(f"\ntest1.model.pre_embedding_size: {test1.model.pre_embedding_size}")
print(f"test2.pre_embedding_size: {test2.pre_embedding_size}")
print(f"是否相同: {test1.model.pre_embedding_size == test2.pre_embedding_size}")
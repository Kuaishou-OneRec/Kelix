import torch
import numpy as np
from copy import deepcopy

# 创建模拟数据
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



# 转换为torch tensor
extended_tokens = torch.tensor(input_data)

print(f"输入tensor形状: {extended_tokens.shape}")
print(f"group_size: 9")

# 创建模拟类以测试两个函数
class MockModel1:
    def __init__(self):
        self.embed_tokens = self.mock_embedding()
        self.pre_embedding_size = None
        self.pre_embedding_tokens = 32000  # 假设的词汇表大小
        
    def mock_embedding(self):
        # 创建一个模拟的embedding层
        class MockEmbedding:
            def __init__(self):
                self.weight = torch.randn(50000, 4096)  # 假设的维度
                
            def __call__(self, x):
                # 模拟embedding操作
                result = torch.randn(*x.shape, 4096)
                # 对于-100，我们返回0向量
                mask = (x == -100)
                result[mask.unsqueeze(-1).expand_as(result)] = 0
                return result
                
        return MockEmbedding()

class MockConfig1:
    def __init__(self):
        class VisionConfig:
            def __init__(self):
                self.n_q_tokens = 8
        self.vision_config = VisionConfig()
        self.vocab_size = 32000

class TestClass1:
    def __init__(self):
        self.config = MockConfig1()
        self.model = MockModel1()
        self.vocab_size = self.config.vocab_size
        
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
            # 这里简化处理，实际应该调用pre_embedding
            stage1_embeds = torch.randn(*vis_emb_input.shape, 4096)
            stage1_embeds = torch.randn(*stage1_embeds.shape)  # 模拟linear
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

class MockModel2:
    def __init__(self):
        self.embed_tokens = self.mock_embedding()
        self.pre_embedding_size = None
        self.pre_embedding_tokens = 32000
        
    def mock_embedding(self):
        class MockEmbedding:
            def __init__(self):
                self.weight = torch.randn(50000, 4096)
                
            def __call__(self, x):
                result = torch.randn(*x.shape, 4096)
                mask = (x == -100)
                result[mask.unsqueeze(-1).expand_as(result)] = 0
                return result
                
        return MockEmbedding()

class TestClass2:
    def __init__(self):
        self.n_q_tokens = 8
        self.embed_tokens = MockModel2().embed_tokens
        self.pre_embedding_size = None
        self.pre_embedding_tokens = 32000
        self.vocab_size = 32000
        
    def infer_id_embs(self, extended_tokens, group_size=None):
        """
        第二个函数的实现
        """
        print(f"extended_tokens3333={extended_tokens}")
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
            # 简化处理
            stage1_embeds = torch.randn(*vis_emb_input.shape, 4096)
            stage1_embeds = torch.randn(*stage1_embeds.shape)
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

# 创建测试实例
test1 = TestClass1()
test2 = TestClass2()

# 运行测试
print("\n=== 测试第一个函数 ===")
output1 = test1.infer_id_embs(extended_tokens, group_size=9)
print(f"输出1形状: {output1.shape}")

print("\n=== 测试第二个函数 ===")
output2 = test2.infer_id_embs(extended_tokens, group_size=9)
print(f"输出2形状: {output2.shape}")
print(f"output1={output1}")
print(f"output2={output2}")


extended_tokens[extended_tokens==-100] = 151681
# 运行测试
print("\n=== 测试第一个函数 ===")
output1 = test1.infer_id_embs(extended_tokens, group_size=9)
print(f"输出1形状: {output1.shape}")

print("\n=== 测试第二个函数 ===")
output2 = test2.infer_id_embs(extended_tokens, group_size=9)
print(f"输出2形状: {output2.shape}")
print(f"output1={output1}")
print(f"output2={output2}")


exit()
# 检查形状是否相同
print(f"\n=== 形状比较 ===")
print(f"输出1和输出2形状是否相同: {output1.shape == output2.shape}")

# 检查数值差异（由于是随机生成的，我们只检查结构）
print(f"\n=== 结构检查 ===")
print(f"is_visual_group的检查:")
# 检查第一个函数中的is_visual_group
first_token1 = extended_tokens.reshape([extended_tokens.shape[0], -1, 9])[:, :, 0].clone()
is_visual_group1 = (first_token1 >= test1.config.vocab_size)
print(f"第一个函数中视觉组数量: {is_visual_group1.sum().item()}")

# 检查第二个函数中的is_visual_group
first_token2 = extended_tokens.reshape([extended_tokens.shape[0], -1, 9])[:, :, 0].clone()
is_visual_group2 = (first_token2 >= test2.vocab_size)
print(f"第二个函数中视觉组数量: {is_visual_group2.sum().item()}")

print(f"视觉组判断是否一致: {torch.all(is_visual_group1 == is_visual_group2)}")

# 检查处理逻辑
print(f"\n=== 关键逻辑检查 ===")
print(f"test1.config.vision_config.n_q_tokens = {test1.config.vision_config.n_q_tokens}")
print(f"test2.n_q_tokens = {test2.n_q_tokens}")
print(f"两者是否一致: {test1.config.vision_config.n_q_tokens == test2.n_q_tokens}")

print(f"\ntest1.config.vocab_size = {test1.config.vocab_size}")
print(f"test2.vocab_size = {test2.vocab_size}")
print(f"两者是否一致: {test1.config.vocab_size == test2.vocab_size}")

# 为了更精确的比较，我们需要确保两个函数使用相同的embedding权重
# 但实际中这是不可能的，因为它们是不同的实例

print(f"\n=== 结论 ===")
print("两个函数在逻辑上是相同的，但有以下区别：")
print("1. 第一个函数通过 self.config.vision_config.n_q_tokens 获取配置")
print("2. 第二个函数直接使用 self.n_q_tokens")
print("3. 第一个函数通过 self.model.embed_tokens 访问embedding层")
print("4. 第二个函数直接使用 self.embed_tokens")
print("5. 属性访问路径不同，但核心逻辑完全一致")
print("\n如果两个类中的相应属性被设置为相同的值，那么输出应该相同。")
print("但在实际运行时，由于embedding层的权重不同，输出值会有差异。")
import torch

# 模拟第一个函数的类和配置
class ModelForFirstFunc:
    def __init__(self, vocab_size, pre_embedding_size=None, pre_embedding_tokens=None, dim=4096):
        self.config = type('Config', (), {
            'vision_config': type('VisionConfig', (), {'n_q_tokens': 8}),  # n_q_tokens+1=9
            'vocab_size': vocab_size
        })()
        self.vocab_size = vocab_size
        self.pre_embedding_size = pre_embedding_size
        self.pre_embedding_tokens = pre_embedding_tokens
        self.dim = dim
        
        # 模拟嵌入层（固定随机种子保证结果可复现）
        torch.manual_seed(42)
        self.embed_tokens = torch.nn.Embedding(vocab_size + 200000, dim)  # 覆盖视觉token范围
        if pre_embedding_size is not None:
            self.pre_embedding = torch.nn.Embedding(pre_embedding_tokens, pre_embedding_size)
            self.pre_embedding_linear = torch.nn.Linear(pre_embedding_size, dim)

class FirstInferClass:
    def __init__(self, vocab_size=150000, pre_embedding_size=None, pre_embedding_tokens=None):
        self.model = ModelForFirstFunc(vocab_size, pre_embedding_size, pre_embedding_tokens)
        self.config = self.model.config
        self.vocab_size = vocab_size

    def infer_id_embs(self, extended_tokens, group_size=None):
        """
        input extended_tokens: batchsize x seqlen x (n_q_tokens + 1)
        output token_inputs_embeds: batchsize x seqlen x (n_q_tokens + 1) x dim
        """
        if group_size is None:
            group_size = self.config.vision_config.n_q_tokens + 1
        
        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        batch_size, compressed_len, dim = extended_tokens.shape
        input_ids_reshaped = extended_tokens

        # 2. 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.config.vocab_size)

        first_token[(first_token>=self.config.vocab_size) | (first_token<0)] = 0  # 把vision tokens 置零
        text_embeds = self.model.embed_tokens(first_token)
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 这里的 0 是为了安全计算，这些计算结果最后会被 mask 掉
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))

        if self.model.pre_embedding_size is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input[(vis_emb_input >= self.model.pre_embedding_tokens) | (vis_emb_input<0)] = 0  #  把text tokens 置零
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

# 模拟第二个函数的类
class SecondInferClass:
    def __init__(self, vocab_size=150000, n_q_tokens=8, pre_embedding_size=None, pre_embedding_tokens=None, dim=4096):
        self.vocab_size = vocab_size
        self.n_q_tokens = n_q_tokens
        self.pre_embedding_size = pre_embedding_size
        self.pre_embedding_tokens = pre_embedding_tokens
        self.dim = dim
        
        # 模拟嵌入层（使用相同随机种子保证权重一致）
        torch.manual_seed(42)
        self.embed_tokens = torch.nn.Embedding(vocab_size + 200000, dim)
        if pre_embedding_size is not None:
            self.pre_embedding = torch.nn.Embedding(pre_embedding_tokens, pre_embedding_size)
            self.pre_embedding_linear = torch.nn.Linear(pre_embedding_size, dim)

    def infer_id_embs(self, extended_tokens, group_size=None):
        """
        input extended_tokens: batchsize x seqlen x (n_q_tokens + 1)
        output token_inputs_embeds: batchsize x seqlen x (n_q_tokens + 1) x dim
        """
        # print(f"extended_tokens3333={extended_tokens}")
        if group_size is None:
            group_size = self.n_q_tokens + 1

        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        input_ids_reshaped = extended_tokens

        # 2. 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.vocab_size)

        first_token[(first_token>=self.config.vocab_size) | (first_token<0)] = 0  # 把vision tokens 置零
        text_embeds = self.embed_tokens(first_token)
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 这里的 0 是为了安全计算，这些计算结果最后会被 mask 掉
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))

        if self.pre_embedding_size is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input[(vis_emb_input >= self.pre_embedding_tokens) | (vis_emb_input<0)] = 0  #  把text tokens 置零
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

# 构造测试输入
def build_test_input():
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
    # 转换为torch tensor，添加batch维度（batchsize=1）
    extended_tokens = torch.tensor(input_data).unsqueeze(0)
    return extended_tokens

# 执行测试
def test_two_functions():
    # 1. 初始化参数（根据常见配置设置，你可根据实际情况调整）
    vocab_size = 150000  # 视觉token起始值151652 > 150000，符合is_visual_group判断逻辑
    pre_embedding_size = None  # 先测试无pre_embedding的情况
    pre_embedding_tokens = None
    group_size = 9
    
    # 2. 构造输入
    extended_tokens = build_test_input()
    print(f"输入形状: {extended_tokens.shape}")  # 应该是 [1, 27, 9]
    
    # 3. 初始化两个类的实例
    first_instance = FirstInferClass(vocab_size, pre_embedding_size, pre_embedding_tokens)
    second_instance = SecondInferClass(vocab_size, n_q_tokens=8, pre_embedding_size=pre_embedding_size, pre_embedding_tokens=pre_embedding_tokens)
    
    # 修复第二个函数中的笔误（self.config.vocab_size → self.vocab_size）
    # 原第二个函数中有一行写错了，这里动态修复
    def corrected_infer_id_embs(self, extended_tokens, group_size=None):
        if group_size is None:
            group_size = self.n_q_tokens + 1

        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        input_ids_reshaped = extended_tokens

        # 2. 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.vocab_size)

        first_token[(first_token>=self.vocab_size) | (first_token<0)] = 0  # 修复：self.config.vocab_size → self.vocab_size
        text_embeds = self.embed_tokens(first_token)
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 这里的 0 是为了安全计算，这些计算结果最后会被 mask 掉
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
    second_instance.infer_id_embs = corrected_infer_id_embs.__get__(second_instance, SecondInferClass)
    
    # 4. 调用两个函数
    output1 = first_instance.infer_id_embs(extended_tokens, group_size=group_size)
    output2 = second_instance.infer_id_embs(extended_tokens, group_size=group_size)
    
    # 5. 对比输出
    print(f"\n第一个函数输出形状: {output1.shape}")
    print(f"第二个函数输出形状: {output2.shape}")
    
    # 检查形状是否一致
    assert output1.shape == output2.shape, "输出形状不一致！"
    print("\n✅ 输出形状一致")
    
    # 检查数值是否完全一致
    is_equal = torch.allclose(output1, output2, atol=1e-6)
    if is_equal:
        print("✅ 输出数值完全一致")
    else:
        print("❌ 输出数值不一致")
        # 找出不一致的位置
        diff_mask = ~torch.isclose(output1, output2, atol=1e-6)
        diff_indices = torch.nonzero(diff_mask)
        print(f"不一致的位置数量: {len(diff_indices)}")
        if len(diff_indices) > 0:
            print(f"第一个不一致的位置: {diff_indices[0].tolist()}")
            print(f"第一个函数对应值: {output1[tuple(diff_indices[0])].item()}")
            print(f"第二个函数对应值: {output2[tuple(diff_indices[0])].item()}")

if __name__ == "__main__":
    test_two_functions()
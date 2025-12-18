import torch
import warnings

# 禁用pynvml警告
warnings.filterwarnings("ignore")

# ====================== 核心配置参数（按你的要求设置） ======================
VOCAB_SIZE = 151936          # 你的vocab_size
PRE_EMBEDDING_TOKENS = 63336 # 你的pre_embedding_tokens
PRE_EMBEDDING_SIZE = 1024    # 可根据你的实际值调整，这里给默认值
EMBED_DIM = 4096             # 嵌入维度，保持和原代码一致
N_Q_TOKENS = 8               # n_q_tokens+1=9，对应group_size=9
# ============================================================================

# 模拟第一个函数的类和配置
class ModelForFirstFunc:
    def __init__(self, vocab_size, device, pre_embedding_size=None, pre_embedding_tokens=None, dim=4096):
        self.config = type('Config', (), {
            'vision_config': type('VisionConfig', (), {'n_q_tokens': N_Q_TOKENS}),
            'vocab_size': vocab_size
        })()
        self.vocab_size = vocab_size
        self.pre_embedding_size = pre_embedding_size
        self.pre_embedding_tokens = pre_embedding_tokens
        self.dim = dim
        self.device = device
        
        # 模拟嵌入层 - 足够大的维度 + 移到指定设备 + padding_idx=0
        torch.manual_seed(42)
        self.embed_tokens = torch.nn.Embedding(300000, dim, padding_idx=0).to(device)  
        
        # 初始化pre_embedding（如果需要）
        if pre_embedding_size is not None and pre_embedding_tokens is not None:
            self.pre_embedding = torch.nn.Embedding(pre_embedding_tokens, pre_embedding_size, padding_idx=0).to(device)
            self.pre_embedding_linear = torch.nn.Linear(pre_embedding_size, dim).to(device)

class FirstInferClass:
    def __init__(self, vocab_size=VOCAB_SIZE, device="cuda", pre_embedding_size=PRE_EMBEDDING_SIZE, pre_embedding_tokens=PRE_EMBEDDING_TOKENS):
        self.device = device
        self.model = ModelForFirstFunc(vocab_size, device, pre_embedding_size, pre_embedding_tokens)
        self.config = self.model.config
        self.vocab_size = vocab_size

    def infer_id_embs(self, extended_tokens, group_size=None):
        """
        input extended_tokens: batchsize x seqlen x (n_q_tokens + 1)
        output token_inputs_embeds: batchsize x seqlen x (n_q_tokens + 1) x dim
        """
        # 确保输入tensor在正确设备上
        extended_tokens = extended_tokens.to(self.device)
        
        if group_size is None:
            group_size = self.config.vision_config.n_q_tokens + 1
        
        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        batch_size, compressed_len, dim = extended_tokens.shape
        input_ids_reshaped = extended_tokens

        # 2. 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.config.vocab_size)

        # 处理负数和超大索引
        first_token = torch.clamp(first_token, min=0)  # 负数置0
        first_token[(first_token >= self.config.vocab_size)] = 0  # 视觉token置零
        text_embeds = self.model.embed_tokens(first_token)
        
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 彻底的安全处理：1. 负数置0 2. 超过嵌入层维度的置0 3. 非视觉组置0
        safe_visual_indices = raw_visual_indices.clone()
        # 第一步：负数全部置0
        safe_visual_indices = torch.clamp(safe_visual_indices, min=0)
        # 第二步：超过嵌入层维度的置0
        embed_max_idx = self.model.embed_tokens.weight.shape[0] - 1
        safe_visual_indices = torch.where(
            safe_visual_indices > embed_max_idx,
            torch.zeros_like(safe_visual_indices),
            safe_visual_indices
        )
        # 第三步：只保留视觉组的索引，非视觉组置0（原逻辑）
        safe_visual_indices = torch.where(mask_expanded_indices, safe_visual_indices, torch.zeros_like(safe_visual_indices))

        if self.model.pre_embedding_size is not None and self.model.pre_embedding_tokens is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input = torch.clamp(vis_emb_input, min=0)  # 负数置0
            vis_emb_input[(vis_emb_input >= self.model.pre_embedding_tokens)] = 0  # 使用你的63336
            stage1_embeds = self.model.pre_embedding(vis_emb_input).detach()
            stage1_embeds = self.model.pre_embedding_linear(stage1_embeds)
            visual_embeds_final = stage1_embeds
        else:
            # 确保输入到embedding的索引绝对安全
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
    def __init__(self, vocab_size=VOCAB_SIZE, device="cuda", n_q_tokens=N_Q_TOKENS, pre_embedding_size=PRE_EMBEDDING_SIZE, pre_embedding_tokens=PRE_EMBEDDING_TOKENS, dim=EMBED_DIM):
        self.vocab_size = vocab_size
        self.n_q_tokens = n_q_tokens
        self.pre_embedding_size = pre_embedding_size
        self.pre_embedding_tokens = pre_embedding_tokens
        self.dim = dim
        self.device = device
        
        # 模拟嵌入层 - 移到指定设备
        torch.manual_seed(42)
        self.embed_tokens = torch.nn.Embedding(300000, dim, padding_idx=0).to(device)
        
        # 初始化pre_embedding（如果需要）
        if pre_embedding_size is not None and pre_embedding_tokens is not None:
            self.pre_embedding = torch.nn.Embedding(pre_embedding_tokens, pre_embedding_size, padding_idx=0).to(device)
            self.pre_embedding_linear = torch.nn.Linear(pre_embedding_size, dim).to(device)

    def infer_id_embs(self, extended_tokens, group_size=None):
        """
        input extended_tokens: batchsize x seqlen x (n_q_tokens + 1)
        output token_inputs_embeds: batchsize x seqlen x (n_q_tokens + 1) x dim
        """
        # 确保输入tensor在正确设备上
        extended_tokens = extended_tokens.to(self.device)
        
        if group_size is None:
            group_size = self.n_q_tokens + 1

        extended_tokens = extended_tokens.reshape([extended_tokens.shape[0], -1, group_size])
        input_ids_reshaped = extended_tokens

        # 2. 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.vocab_size)

        # 处理负数和超大索引
        first_token = torch.clamp(first_token, min=0)  # 负数置0
        first_token[(first_token >= self.vocab_size)] = 0  # 视觉token置零
        text_embeds = self.embed_tokens(first_token)
        
        raw_visual_indices = input_ids_reshaped[:, :, :-1] if group_size > 1 else input_ids_reshaped
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 彻底的安全处理
        safe_visual_indices = raw_visual_indices.clone()
        # 第一步：负数全部置0
        safe_visual_indices = torch.clamp(safe_visual_indices, min=0)
        # 第二步：超过嵌入层维度的置0
        embed_max_idx = self.embed_tokens.weight.shape[0] - 1
        safe_visual_indices = torch.where(
            safe_visual_indices > embed_max_idx,
            torch.zeros_like(safe_visual_indices),
            safe_visual_indices
        )
        # 第三步：只保留视觉组的索引，非视觉组置0
        safe_visual_indices = torch.where(mask_expanded_indices, safe_visual_indices, torch.zeros_like(safe_visual_indices))

        if self.pre_embedding_size is not None and self.pre_embedding_tokens is not None:
            vis_emb_input = (safe_visual_indices % self.vocab_size).clone()
            vis_emb_input = torch.clamp(vis_emb_input, min=0)  # 负数置0
            vis_emb_input[(vis_emb_input >= self.pre_embedding_tokens)] = 0  # 使用你的63336
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
def build_test_input(device="cuda"):
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
    # 转换为torch tensor（long类型）并移到指定设备
    extended_tokens = torch.tensor(input_data, dtype=torch.long).unsqueeze(0).to(device)
    return extended_tokens

# 执行测试
def test_two_functions():
    # 1. 设置设备（强制使用CUDA）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 2. 初始化参数（使用你的配置）
    vocab_size = VOCAB_SIZE
    pre_embedding_size = PRE_EMBEDDING_SIZE  # 设为None则使用embed_tokens分支
    pre_embedding_tokens = PRE_EMBEDDING_TOKENS
    group_size = 9
    
    # 3. 构造输入（直接移到GPU）
    extended_tokens = build_test_input(device)
    print(f"输入形状: {extended_tokens.shape}")
    print(f"输入设备: {extended_tokens.device}")
    print(f"使用配置: vocab_size={vocab_size}, pre_embedding_tokens={pre_embedding_tokens}")
    
    # 4. 初始化两个类的实例（指定设备和你的参数）
    first_instance = FirstInferClass(
        vocab_size=vocab_size, 
        device=device, 
        pre_embedding_size=pre_embedding_size,
        pre_embedding_tokens=pre_embedding_tokens
    )
    second_instance = SecondInferClass(
        vocab_size=vocab_size, 
        device=device, 
        n_q_tokens=N_Q_TOKENS,
        pre_embedding_size=pre_embedding_size,
        pre_embedding_tokens=pre_embedding_tokens
    )
    
    # 5. 调用两个函数（禁用梯度）
    with torch.no_grad():
        output1 = first_instance.infer_id_embs(extended_tokens, group_size=group_size)
        output2 = second_instance.infer_id_embs(extended_tokens, group_size=group_size)
    
    # 6. 对比输出
    print(f"\n第一个函数输出形状: {output1.shape}")
    print(f"第二个函数输出形状: {output2.shape}")
    print(f"输出设备: {output1.device}")
    
    # 检查形状是否一致
    assert output1.shape == output2.shape, "输出形状不一致！"
    print("\n✅ 输出形状一致")
    
    # 检查数值是否完全一致
    is_equal = torch.allclose(output1, output2, atol=1e-6)
    if is_equal:
        print("✅ 输出数值完全一致")
    else:
        print("❌ 输出数值不一致")
        # 找出不一致的位置（只检查前5个）
        diff_mask = ~torch.isclose(output1, output2, atol=1e-6)
        diff_indices = torch.nonzero(diff_mask, as_tuple=False)[:5]
        print(f"前5个不一致的位置:")
        for idx in diff_indices:
            idx_list = idx.tolist()
            val1 = output1[tuple(idx_list)].item()
            val2 = output2[tuple(idx_list)].item()
            print(f"位置 {idx_list}: 第一个函数={val1:.6f}, 第二个函数={val2:.6f}")

if __name__ == "__main__":
    # 可选：如果你想测试无pre_embedding的情况，取消下面注释
    # PRE_EMBEDDING_SIZE = None
    # PRE_EMBEDDING_TOKENS = None
    
    test_two_functions()
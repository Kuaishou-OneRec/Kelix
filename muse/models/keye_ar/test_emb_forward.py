import torch
import warnings
warnings.filterwarnings("ignore")

# ====================== 配置参数 ======================
VOCAB_SIZE = 151936
PRE_EMBEDDING_TOKENS = 63336
PRE_EMBEDDING_SIZE = None  # 设为None走embed_tokens分支
N_Q_TOKENS = 8
GROUP_SIZE = 9
EMBED_DIM = 4096
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 模拟第一个函数的类 ======================
class FirstClass:
    def __init__(self):
        self.config = type('Config', (), {
            'vocab_size': VOCAB_SIZE,
            'vision_config': type('VisionConfig', (), {'n_q_tokens': N_Q_TOKENS})
        })()
        self.vocab_size = VOCAB_SIZE
        self.model = type('Model', (), {
            'pre_embedding_size': PRE_EMBEDDING_SIZE,
            'pre_embedding_tokens': PRE_EMBEDDING_TOKENS,
            'embed_tokens': torch.nn.Embedding(300000, EMBED_DIM, padding_idx=0).to(DEVICE)
        })()
        # 固定随机种子保证权重一致
        torch.manual_seed(42)

    def core_logic(self, extended_tokens):
        batch_size = extended_tokens.shape[0]
        compressed_len = extended_tokens.shape[1] // GROUP_SIZE
        input_ids_reshaped = extended_tokens.reshape([batch_size, compressed_len, GROUP_SIZE])

        # 识别视觉组 Mask
        first_token = input_ids_reshaped[:, :, 0].clone()
        is_visual_group = (first_token >= self.config.vocab_size)

        first_token[(first_token>=self.config.vocab_size) | (first_token<0)] = 0
        text_embeds = self.model.embed_tokens(first_token)
        raw_visual_indices = input_ids_reshaped[:, :, 1:]
        mask_expanded_indices = is_visual_group.unsqueeze(-1).expand_as(raw_visual_indices)
        
        # 安全索引处理
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))
        # 负数和超大索引安全处理
        safe_visual_indices = torch.clamp(safe_visual_indices, min=0)
        safe_visual_indices = torch.where(safe_visual_indices > 299999, torch.zeros_like(safe_visual_indices), safe_visual_indices)

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
        text_embeds = text_embeds[:,:,None].repeat_interleave(self.config.vision_config.n_q_tokens, dim=2)
        token_inputs_embeds = torch.where(mask_final[:, :, None, :], visual_embeds_final, text_embeds)
        return token_inputs_embeds

# ====================== 模拟第二个函数的类 ======================
class SecondClass:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.n_q_tokens = N_Q_TOKENS
        self.pre_embedding_size = PRE_EMBEDDING_SIZE
        self.pre_embedding_tokens = PRE_EMBEDDING_TOKENS
        # 固定随机种子保证权重和第一个类一致
        torch.manual_seed(42)
        self.embed_tokens = torch.nn.Embedding(300000, EMBED_DIM, padding_idx=0).to(DEVICE)

    def core_logic(self, extended_tokens, group_size=None):
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

        # 安全索引处理
        safe_visual_indices = torch.where(mask_expanded_indices, raw_visual_indices, torch.zeros_like(raw_visual_indices))
        # 负数和超大索引安全处理
        safe_visual_indices = torch.clamp(safe_visual_indices, min=0)
        safe_visual_indices = torch.where(safe_visual_indices > 299999, torch.zeros_like(safe_visual_indices), safe_visual_indices)

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

# ====================== 构造输入并测试 ======================
def main():
    # 1. 构造输入
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
    extended_tokens = torch.tensor(input_data, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 2. 初始化实例
    first_inst = FirstClass()
    second_inst = SecondClass()

    # 3. 执行核心逻辑
    with torch.no_grad():
        output1 = first_inst.core_logic(extended_tokens)
        output2 = second_inst.core_logic(extended_tokens, group_size=GROUP_SIZE)

    # 4. 验证结果是否相等
    shape_equal = output1.shape == output2.shape
    value_equal = torch.allclose(output1, output2, atol=1e-6)
    
    print(f"形状是否一致: {shape_equal}")
    print(f"数值是否一致: {value_equal}")
    print(f"最终结论: {'两个代码输出相等' if shape_equal and value_equal else '两个代码输出不相等'}")

if __name__ == "__main__":
    main()
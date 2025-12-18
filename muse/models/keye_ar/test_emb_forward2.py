import torch
import warnings
warnings.filterwarnings("ignore")

# ====================== 核心配置 ======================
VOCAB_SIZE = 151936 + 65536  # 217472
PRE_EMBEDDING_SIZE = None
PRE_EMBEDDING_TOKENS = 63336
N_Q_TOKENS = 8
GROUP_SIZE = 9
EMBED_DIM = 4096
IMAGE_TOKEN_ID = 151655  # 从input_ids中推断的image_token_id
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 共享的嵌入层（保证两个代码使用同一套权重） ======================
class SharedEmbedding:
    def __init__(self):
        torch.manual_seed(42)
        self.embed_tokens = torch.nn.Embedding(300000, EMBED_DIM, padding_idx=0).to(DEVICE)
        self.pre_embedding = None
        self.pre_embedding_linear = None

shared_emb = SharedEmbedding()

# ====================== 第一个代码：Forward类 ======================
class FirstModel:
    def __init__(self):
        self.vocab_size = VOCAB_SIZE
        self.n_q_tokens = N_Q_TOKENS
        self.pre_embedding_size = PRE_EMBEDDING_SIZE
        self.pre_embedding_tokens = PRE_EMBEDDING_TOKENS
        self.embed_tokens = shared_emb.embed_tokens  # 共享嵌入层

    def forward(self, extended_tokens, aggregation=True):
        embeddings = self._get_token_embeddings(extended_tokens)
        if aggregation:
            aggregated_embeddings = self._embedding_aggregation(extended_tokens, embeddings)
            return aggregated_embeddings
        else:
            return embeddings

    def _get_token_embeddings(self, extended_tokens, group_size=None):
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

    def _embedding_aggregation(self, extended_tokens, embeddings):
        """模拟聚合逻辑：对n_q_tokens维度求和（和第二个代码的sum(1)对齐）"""
        # embeddings shape: [batch_size, seqlen, n_q_tokens, dim]
        # 求和后 shape: [batch_size, seqlen, dim]
        aggregated = embeddings.sum(dim=2)
        return aggregated

# ====================== 第二个代码：基准逻辑 ======================
class SecondModel:
    def __init__(self):
        self.config = type('Config', (), {
            'new_table': True,
            'vision_config': type('VisionConfig', (), {'n_q_tokens': N_Q_TOKENS}),
            'image_token_id': IMAGE_TOKEN_ID
        })()
        self.vocab_size = VOCAB_SIZE
        self.pre_embedding_size = PRE_EMBEDDING_SIZE
        self.embed_tokens = shared_emb.embed_tokens  # 共享嵌入层
        self.pre_embedding = None
        self.pre_embedding_linear = None

    def get_inputs_embeds(self, input_ids, input_image_ids):
        # 核心逻辑
        inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "new_table", False):
            if input_image_ids is None:
                if input_ids is not None:
                    input_image_ids = torch.zeros_like(input_ids[..., :0])
                else:
                    input_image_ids = torch.zeros_like(inputs_embeds[..., :0, 0]).long()

            if self.pre_embedding_size is not None:
                input_image_embeds = self.pre_embedding(input_image_ids % self.embed_tokens.num_embeddings).detach()
                input_image_embeds = self.pre_embedding_linear(input_image_embeds)
            else:
                input_image_embeds = self.embed_tokens(input_image_ids)

            if input_image_ids.numel():
                batch, _ = input_image_ids.shape
                input_image_embeds = input_image_embeds.view(batch, self.config.vision_config.n_q_tokens, -1).sum(1)
                mask = (input_ids == self.config.image_token_id)
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)
                input_image_embeds = input_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, input_image_embeds)
        
        return inputs_embeds

# ====================== 构造输入数据 ======================
def build_test_data():
    # 第一个模型的输入
    extended_tokens = torch.tensor([[151645, 151681, 151681, 151681, 151681, 151681, 151681, 151681,
            151681],
           [   198, 151681, 151681, 151681, 151681, 151681, 151681, 151681,
            151681],
           [151644, 151681, 151681, 151681, 151681, 151681, 151681, 151681,
            151681],
           [   872, 151681, 151681, 151681, 151681, 151681, 151681, 151681,
            151681],
           [   198, 151681, 151681, 151681, 151681, 151681, 151681, 151681,
            151681],
           [151652, 151681, 151681, 151681, 151681, 151681, 151681, 151681,
            151681],
           [158989, 163361, 172176, 176543, 191772, 198720, 201995, 209754,
            151681],
           [158723, 161428, 172475, 183267, 189182, 195380, 203676, 215606,
            151681],
           [152707, 161925, 170851, 182049, 190989, 200537, 203905, 211502,
            151681],
           [155872, 161428, 171248, 181246, 186400, 195380, 203676, 210683,
            151681],
           [157266, 164215, 175621, 180185, 188326, 197081, 208022, 216938,
            151681],
           [152940, 160320, 176261, 181962, 188078, 198887, 203676, 212428,
            151681],
           [152707, 167798, 174112, 182049, 187728, 194661, 206012, 214532,
            151681],
           [156614, 166166, 172568, 178561, 190379, 197759, 207621, 212810,
            151681],
           [152707, 160713, 172989, 179118, 192749, 195380, 207734, 216938,
            151681],
           [159674, 161952, 169345, 181686, 191525, 195874, 204839, 210086,
            151681],
           [152707, 161925, 171383, 182049, 190989, 200363, 201154, 214532,
            151681],
           [156614, 163115, 172568, 182049, 190379, 196183, 206012, 213994,
            151681],
           [159418, 161971, 171135, 183504, 189079, 199763, 204210, 213800,
            151681],
           [153068, 166166, 174479, 178561, 187447, 199763, 204210, 214334,
            151681],
           [152707, 161925, 174112, 182049, 187728, 199126, 201204, 214532,
            151681],
           [153068, 160713, 171520, 178561, 192513, 195874, 203905, 211502,
            151681],
           [151653, 151681, 151681, 151681, 151681, 151681, 151681, 151681,
            151681]]).long().unsqueeze(0).to(DEVICE)

    # 第二个模型的输入
    input_ids = torch.tensor([[151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655,
           151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
           151655, 151655, 151655, 151655, 151653]]).long().to(DEVICE)
    
    input_image_ids = torch.tensor([[158989, 163361, 172176, 176543, 191772, 198720, 201995, 209754],
           [158723, 161428, 172475, 183267, 189182, 195380, 203676, 215606],
           [152707, 161925, 170851, 182049, 190989, 200537, 203905, 211502],
           [155872, 161428, 171248, 181246, 186400, 195380, 203676, 210683],
           [157266, 164215, 175621, 180185, 188326, 197081, 208022, 216938],
           [152940, 160320, 176261, 181962, 188078, 198887, 203676, 212428],
           [152707, 167798, 174112, 182049, 187728, 194661, 206012, 214532],
           [156614, 166166, 172568, 178561, 190379, 197759, 207621, 212810],
           [152707, 160713, 172989, 179118, 192749, 195380, 207734, 216938],
           [159674, 161952, 169345, 181686, 191525, 195874, 204839, 210086],
           [152707, 161925, 171383, 182049, 190989, 200363, 201154, 214532],
           [156614, 163115, 172568, 182049, 190379, 196183, 206012, 213994],
           [159418, 161971, 171135, 183504, 189079, 199763, 204210, 213800],
           [153068, 166166, 174479, 178561, 187447, 199763, 204210, 214334],
           [152707, 161925, 174112, 182049, 187728, 199126, 201204, 214532],
           [153068, 160713, 171520, 178561, 192513, 195874, 203905, 211502]]).long().to(DEVICE)
    
    return extended_tokens, input_ids, input_image_ids

# ====================== 主测试逻辑 ======================
def main():
    # 1. 初始化模型和数据
    extended_tokens, input_ids, input_image_ids = build_test_data()
    model1 = FirstModel()
    model2 = SecondModel()

    # 2. 执行前向计算（禁用梯度）
    with torch.no_grad():
        aggregated_embeddings = model1.forward(extended_tokens, aggregation=True)
        inputs_embeds = model2.get_inputs_embeds(input_ids, input_image_ids)

    # 3. 调整形状以匹配（保证维度一致后比较）
    # aggregated_embeddings: [1, 23, 4096]
    # inputs_embeds: [1, 23, 4096]
    # 直接比较数值一致性
    is_close = torch.allclose(aggregated_embeddings, inputs_embeds, atol=1e-6, rtol=1e-6)
    
    # 4. 输出最终结果
    print(f"aggregated_embeddings 和 inputs_embeds 是否一致: {is_close}")

if __name__ == "__main__":
    main()
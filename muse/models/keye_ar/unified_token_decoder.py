import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from muse.models.base import Model
from muse.layers.transformer import TransformerDecoder, TransformerSelfAttentionLayer
from muse.layers.attention import MultiHeadAttention
from muse.layers.feed_forward import FeedForward
from muse.layers.layer_norm import Fp32LayerNorm
from muse.layers.position_embeddings import LlamaRotaryPositionalEmbeddings
from muse.config.model_config import UnifiedTokenDecoderConfig


class UnifiedTokenDecoder(Model):
    def __init__(self, 
                 config: UnifiedTokenDecoderConfig,
                 token_embedding: Optional[nn.Embedding] = None,
                 lm_head: Optional[nn.Linear] = None,
                 infer_id_embs_fn = None):

                
        super().__init__(config)
        
        # 从config中提取参数
        vocab_size = config.vocab_size
        max_length = config.max_length
        max_pos_length = config.max_pos_length
        d_model = config.d_model
        eos_token = config.eos_token
        nhead = config.nhead
        num_layers = config.num_layers
        dim_feedforward = config.dim_feedforward
        use_gradient_checkpointing = config.use_gradient_checkpointing
        input_dim = config.input_dim
        reduce = config.reduce
        attention_function = config.attention_function
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.max_pos_length = max_pos_length
        self.eos_token = eos_token
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_dim = input_dim
        self.reduce = reduce
        self.lm_head = lm_head
        self.infer_id_embs_fn = infer_id_embs_fn
        self.final_norm = nn.LayerNorm(d_model) # unused

        # 检查head_dim合法性
        head_dim = d_model // nhead
        assert head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        # Token Embedding层
        if token_embedding is not None:
            self.token_embedding = token_embedding
            assert token_embedding.embedding_dim == d_model, "Embedding dimension must match d_model"
            assert token_embedding.num_embeddings == vocab_size, "Embedding vocab size must match vocab_size"
        # else:
        #     self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码 - 添加可训练的位置编码以匹配原始模型
        self.position_embedding = nn.Embedding(self.max_pos_length, d_model)
        
        # 创建解码器层
        layers = []
        for _ in range(num_layers):
            # 创建注意力层 - 修改为支持偏置参数
            self_attn = MultiHeadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                num_kv_heads=nhead,
                head_dim=head_dim,
                q_proj=nn.Linear(d_model, nhead * head_dim, bias=True),  # 修改为True
                k_proj=nn.Linear(d_model, nhead * head_dim, bias=True),  # 修改为True
                v_proj=nn.Linear(d_model, nhead * head_dim, bias=True),  # 修改为True
                output_proj=nn.Linear(nhead * head_dim, d_model, bias=True),  # 修改为True
                pos_embeddings=None,  # 移除RoPE，使用可训练位置编码
                kv_cache=None,
                max_seq_len=max_length,
                is_causal=True,
                attn_dropout=0.0,
                attention_function=attention_function
            )
            
            # 创建前馈网络 - 修改为支持偏置参数
            mlp = FeedForward(
                gate_proj=nn.Linear(d_model, dim_feedforward, bias=False),
                up_proj=nn.Linear(d_model, dim_feedforward, bias=False),
                down_proj=nn.Linear(dim_feedforward, d_model, bias=False),
                activation=nn.SiLU()
            )
            
            # 创建Transformer层
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=Fp32LayerNorm(d_model, eps=1e-5),
                mlp_norm=Fp32LayerNorm(d_model, eps=1e-5)
            )
            layers.append(layer)
        
        # 创建输入/输出线性层
        if not reduce:
            self.input_linear = nn.Identity()
            self.output_linear = nn.Identity()
        else:
            assert input_dim is not None, "input_dim must be provided when reduce=True"
            self.input_linear = nn.Linear(input_dim, d_model)
            self.output_linear = nn.Linear(d_model, input_dim)
        
        # 创建Transformer解码器 - 修改为使用nn.Identity()作为tok_embeddings
        self.transformer = TransformerDecoder(
            tok_embeddings=nn.Identity(),  # 修改为Identity，直接接收embeddings
            layers=layers,
            max_seq_len=max_length,
            num_heads=nhead,
            head_dim=head_dim,
            norm=nn.Identity(),  # 修改为Identity，直接接收normed_output
            output=nn.Identity()
        )
    
    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播（输入为embedding）
        Args:
            x_emb: (Batch, Seq_Len, d_model)
        Returns:
            out: (Batch, Seq_Len, d_model) 或 (Batch, Seq_Len, vocab_size) 如果lm_head不为None
        """
        batch_size, seq_len, _ = x_emb.shape
        
        if not self.reduce:
            x_emb0 = 0
        else:
            x_emb0 = x_emb
        
        # 输入线性层
        x_emb = self.input_linear(x_emb)
        
        # 添加位置编码（匹配原始模型）
        positions = torch.arange(seq_len, device=x_emb.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x_emb = x_emb + pos_emb
        
        # 前向传播 - 修改为传入tokens=None，input_embeds=x_emb
        output = self.transformer(tokens=None, input_embeds=x_emb)

        # 输出线性层和残差连接（修复：移除条件判断，始终应用残差连接）
        output = self.output_linear(output)
        output = output + x_emb0

        # 应用lm_head（如果存在）
        if self.lm_head is not None:
            output = self.lm_head(output)
        
        return output
    
    def forward_with_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        完善的辅助方法：支持输入token ID直接前向传播，可选择使用infer_id_embs_fn
        Args:
            tokens: (Batch, Seq_Len) token ID序列
        Returns:
            out: (Batch, Seq_Len, d_model) 或 (Batch, Seq_Len, vocab_size) 如果lm_head不为None
        """
        # 如果提供了infer_id_embs_fn，则使用它来获取token embeddings
        if self.infer_id_embs_fn is not None:
            x_emb = self.infer_id_embs_fn(tokens)
        else:
            # 否则使用标准的token embedding
            x_emb = self.token_embedding(tokens)
        
        # 通过主forward方法传递
        output = self.forward(x_emb)
        return output
    
    def generate(self, input_ids: Optional[torch.Tensor] = None, 
                 tokens: Optional[torch.Tensor] = None,
                 input_embeddings: Optional[torch.Tensor] = None, 
                 max_new_tokens: int = 50, 
                 temperature: float = 1.0, 
                 top_k: Optional[int] = None, 
                 top_p: Optional[float] = None,
                 do_sample: bool = False, 
                 pad_token_id: Optional[int] = None,
                 return_logits: bool = True,
                 only_last: bool = False) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        生成函数：统一使用input_embeddings作为输入（prefill后也保持embedding输入）
        """
        self.eval()

        if tokens is not None:
            input_ids = tokens
        
        # 检查输入合法性
        if (input_ids is None and input_embeddings is None) or \
           (input_ids is not None and input_embeddings is not None):
            raise ValueError("Must provide either input_ids or input_embeddings, not both")
        
        with torch.no_grad():
            # 1. 统一初始输入为embeddings（无论输入是ids还是embeddings）
            if input_ids is not None:
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
                generated_ids = input_ids.clone()
                
                # 从input_ids生成初始embeddings
                if self.infer_id_embs_fn is not None:
                    current_embeddings = self.infer_id_embs_fn(input_ids)
                else:
                    current_embeddings = self.token_embedding(input_ids)
                
            else:  # input_embeddings is not None
                batch_size, seq_len, emb_dim = input_embeddings.shape
                device = input_embeddings.device
                current_embeddings = input_embeddings.clone()
                generated_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
            
            # 初始化logits存储
            logits_list = []
            
            # 2. 自回归生成循环（始终用embeddings输入）
            for _ in range(max_new_tokens):
                # 序列长度限制检查
                if current_embeddings.shape[1] >= self.max_length:
                    break
                
                # 前向传播
                logits = self.forward(current_embeddings)
                
                next_token_logits = logits[:, -1, :]
                
                # 保存logits
                if return_logits:
                    logits_list.append(next_token_logits)
                
                # 采样策略处理
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Top-k过滤
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                # Top-p过滤
                if top_p is not None and top_p < 1.0:
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 0] = 0  # 保留至少一个token
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    for batch_idx in range(batch_size):
                        next_token_logits[batch_idx, indices_to_remove[batch_idx]] = -float('Inf')
                
                # 生成下一个token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # 更新生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # 生成新token的embedding并更新输入embeddings
                if self.infer_id_embs_fn is not None:
                    next_token_emb = self.infer_id_embs_fn(next_token)
                else:
                    next_token_emb = self.token_embedding(next_token)
                
                current_embeddings = torch.cat([current_embeddings, next_token_emb], dim=1)
                
                # 检查EOS停止条件
                if self.eos_token is not None:
                    if next_token[-1, 0] == self.eos_token:
                        break
            
            # 处理返回结果
            if only_last:
                generated_ids = generated_ids[..., -1:]
                logits_list = logits_list[-1:]
                
            if return_logits and logits_list:
                logits_tensor = torch.stack(logits_list, dim=1)
                return generated_ids, logits_tensor
            
            return generated_ids
    
    @staticmethod
    def convert_hf_state_dict(state_dict: dict, reduce_mode: bool = False) -> dict:
        """
        将原始模型的状态字典转换为新模型的状态字典
        Args:
            state_dict: 原始模型的状态字典
            reduce_mode: 是否为reduce模式（影响output_linear的处理）
        Returns:
            converted_state_dict: 转换后的状态字典
        """
        converted_state_dict = {}
        skipped_keys = []
        converted_count = 0
        total_count = len(state_dict)
        
        # print(f"原始模型状态字典键数: {total_count}")
        
        for key, value in state_dict.items():
            new_key = None
            
            # 1. 处理Token Embedding层 - 修改为只复制到token_embedding.weight
            if key == "token_embedding.weight":
                # 现在只需要复制到token_embedding.weight，因为TransformerDecoder使用Identity
                converted_state_dict["token_embedding.weight"] = value
                converted_count += 1
                continue
            
            # 2. 处理输入/输出线性层
            elif key == "input_linear.weight":
                new_key = "input_linear.weight"
            elif key == "input_linear.bias":
                new_key = "input_linear.bias"
            elif key == "output_linear.weight":
                # 关键修复：当lm_head为None且reduce=False时，原始模型output_linear是Identity
                # 新模型transformer.output也是Identity，不需要映射
                if not reduce_mode:
                    # 跳过这个键，因为新模型不需要它
                    skipped_keys.append(key)
                    continue
                else:
                    # reduce=True时，映射到output_linear.weight
                    new_key = "output_linear.weight"
            elif key == "output_linear.bias":
                new_key = "output_linear.bias"
            
            # 3. 处理LM Head
            elif key == "lm_head.weight":
                new_key = "lm_head.weight"
            elif key == "lm_head.bias":
                new_key = "lm_head.bias"
            
            # 4. 处理Decoder层
            elif key.startswith("layers."):
                # 提取层索引
                layer_parts = key.split(".")
                layer_idx = layer_parts[1]
                
                # 4.1 处理层归一化
                if ".norm1.weight" in key:
                    new_key = f"transformer.layers.{layer_idx}.sa_norm.weight"
                elif ".norm1.bias" in key:
                    new_key = f"transformer.layers.{layer_idx}.sa_norm.bias"
                elif ".norm2.weight" in key:
                    new_key = f"transformer.layers.{layer_idx}.mlp_norm.weight"
                elif ".norm2.bias" in key:
                    new_key = f"transformer.layers.{layer_idx}.mlp_norm.bias"
                
                # 4.2 处理注意力层 - 修复：转换权重和偏置
                elif ".qkv_proj.weight" in key:
                    # 拆分qkv_proj为q_proj, k_proj, v_proj（只转换权重）
                    d_model = value.shape[0]  # 获取d_model维度
                    q_weight, k_weight, v_weight = torch.chunk(value, 3, dim=0)
                    converted_state_dict[f"transformer.layers.{layer_idx}.attn.q_proj.weight"] = q_weight
                    converted_state_dict[f"transformer.layers.{layer_idx}.attn.k_proj.weight"] = k_weight
                    converted_state_dict[f"transformer.layers.{layer_idx}.attn.v_proj.weight"] = v_weight
                    converted_count += 1
                    continue
                elif ".qkv_proj.bias" in key:
                    # 拆分qkv_proj.bias为q_proj.bias, k_proj.bias, v_proj.bias
                    d_model = value.shape[0]  # 获取d_model维度
                    q_bias, k_bias, v_bias = torch.chunk(value, 3, dim=0)
                    converted_state_dict[f"transformer.layers.{layer_idx}.attn.q_proj.bias"] = q_bias
                    converted_state_dict[f"transformer.layers.{layer_idx}.attn.k_proj.bias"] = k_bias
                    converted_state_dict[f"transformer.layers.{layer_idx}.attn.v_proj.bias"] = v_bias
                    converted_count += 1
                    continue
                elif ".out_proj.weight" in key:
                    new_key = f"transformer.layers.{layer_idx}.attn.output_proj.weight"
                elif ".out_proj.bias" in key:
                    new_key = f"transformer.layers.{layer_idx}.attn.output_proj.bias"
                
                # 4.3 处理MLP层
                elif ".mlp.gate_proj.weight" in key:
                    new_key = f"transformer.layers.{layer_idx}.mlp.w1.weight"
                elif ".mlp.up_proj.weight" in key:
                    new_key = f"transformer.layers.{layer_idx}.mlp.w3.weight"
                elif ".mlp.down_proj.weight" in key:
                    new_key = f"transformer.layers.{layer_idx}.mlp.w2.weight"
            
            # 5. 处理最终归一化
            elif key == "final_norm.weight":
                new_key = "final_norm.weight"
            elif key == "final_norm.bias":
                new_key = "final_norm.bias"
            
            # 6. 处理位置编码（现在模型有位置编码层，直接映射）
            elif key == "position_embedding.weight":
                new_key = "position_embedding.weight"
            
            # 如果找到了新键名
            if new_key is not None:
                converted_state_dict[new_key] = value
                converted_count += 1
            else:
                # 其他键跳过
                skipped_keys.append(key)
        
        # 关键修复：当reduce=False且lm_head为None时，不需要创建transformer.output.weight
        # 因为transformer.output是nn.Identity()
        if not reduce_mode and "transformer.output.weight" not in converted_state_dict:
            # 只有当原始模型有lm_head时，才需要创建transformer.output.weight
            if "lm_head.weight" in state_dict:
                # 从token_embedding.weight获取d_model维度
                d_model = state_dict["token_embedding.weight"].shape[1]
                # 创建单位矩阵作为transformer.output.weight
                identity_matrix = torch.eye(d_model)
                converted_state_dict["transformer.output.weight"] = identity_matrix
                print(f"  为reduce=False模式创建单位矩阵transformer.output.weight，维度: {d_model}")
            else:
                print(f"  reduce=False且lm_head=None，transformer.output使用Identity，无需权重")
        
        # 打印转换统计信息
        print(f"转换状态字典统计:")
        print(f"  总键数: {total_count}")
        print(f"  转换成功: {converted_count}")
        print(f"  跳过键数: {len(skipped_keys)}")
        if skipped_keys:
            print(f"  跳过的键: {skipped_keys}")
        
        return converted_state_dict

    @staticmethod
    def revert_hf_state_dict(state_dict: dict, reduce_mode: bool = False) -> dict:
        """
        将新模型的状态字典转换回原始模型的状态字典（convert_hf_state_dict的逆操作）
        Args:
            state_dict: 新模型的状态字典
            reduce_mode: 是否为reduce模式（影响output_linear的处理）
        Returns:
            reverted_state_dict: 转换回的原始模型状态字典
        """
        reverted_state_dict = {}
        skipped_keys = []
        reverted_count = 0
        total_count = len(state_dict)
        
        print(f"新模型状态字典键数: {total_count}")
        
        for key, value in state_dict.items():
            old_key = None
            
            # 1. 处理Token Embedding层
            if key == "token_embedding.weight":
                old_key = "token_embedding.weight"
                reverted_state_dict[old_key] = value
                reverted_count += 1
                continue
            
            # 2. 处理输入/输出线性层
            elif key == "input_linear.weight":
                old_key = "input_linear.weight"
            elif key == "input_linear.bias":
                old_key = "input_linear.bias"
            elif key == "output_linear.weight":
                # 根据reduce_mode决定是否映射
                if reduce_mode:
                    old_key = "output_linear.weight"
                else:
                    # reduce=False时，跳过这个键
                    skipped_keys.append(key)
                    continue
            elif key == "output_linear.bias":
                old_key = "output_linear.bias"
            
            # 3. 处理LM Head
            elif key == "lm_head.weight":
                old_key = "lm_head.weight"
            elif key == "lm_head.bias":
                old_key = "lm_head.bias"
            elif key == "final_norm.weight":
                old_key = "final_norm.weight"
            elif key == "final_norm.bias":
                old_key = "final_norm.bias"
            
            # 4. 处理Decoder层
            elif key.startswith("transformer.layers."):
                # 提取层索引
                layer_parts = key.split(".")
                layer_idx = layer_parts[2]
                
                # 4.1 处理层归一化
                if ".sa_norm.weight" in key:
                    old_key = f"layers.{layer_idx}.norm1.weight"
                elif ".sa_norm.bias" in key:
                    old_key = f"layers.{layer_idx}.norm1.bias"
                elif ".mlp_norm.weight" in key:
                    old_key = f"layers.{layer_idx}.norm2.weight"
                elif ".mlp_norm.bias" in key:
                    old_key = f"layers.{layer_idx}.norm2.bias"
                
                # 4.2 处理注意力层
                elif ".attn.q_proj.weight" in key:
                    # 收集q,k,v权重，稍后合并
                    q_weight_key = key
                    k_weight_key = key.replace(".q_proj.", ".k_proj.")
                    v_weight_key = key.replace(".q_proj.", ".v_proj.")
                    
                    # 检查是否都有对应的k,v权重
                    if k_weight_key in state_dict and v_weight_key in state_dict:
                        q_weight = state_dict[q_weight_key]
                        k_weight = state_dict[k_weight_key]
                        v_weight = state_dict[v_weight_key]
                        
                        # 合并为qkv_proj.weight
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                        old_key = f"layers.{layer_idx}.qkv_proj.weight"
                        reverted_state_dict[old_key] = qkv_weight
                        reverted_count += 1
                        
                        # 跳过k_proj和v_proj的处理
                        skipped_keys.extend([k_weight_key, v_weight_key])
                        continue
                    else:
                        # 如果缺少对应的权重，跳过
                        skipped_keys.append(key)
                        continue
                elif ".attn.q_proj.bias" in key:
                    # 收集q,k,v偏置，稍后合并
                    q_bias_key = key
                    k_bias_key = key.replace(".q_proj.", ".k_proj.")
                    v_bias_key = key.replace(".q_proj.", ".v_proj.")
                    
                    # 检查是否都有对应的k,v偏置
                    if k_bias_key in state_dict and v_bias_key in state_dict:
                        q_bias = state_dict[q_bias_key]
                        k_bias = state_dict[k_bias_key]
                        v_bias = state_dict[v_bias_key]
                        
                        # 合并为qkv_proj.bias
                        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                        old_key = f"layers.{layer_idx}.qkv_proj.bias"
                        reverted_state_dict[old_key] = qkv_bias
                        reverted_count += 1
                        
                        # 跳过k_proj和v_proj的处理
                        skipped_keys.extend([k_bias_key, v_bias_key])
                        continue
                    else:
                        # 如果缺少对应的偏置，跳过
                        skipped_keys.append(key)
                        continue
                elif ".attn.k_proj." in key or ".attn.v_proj." in key:
                    # 这些权重已经在上面处理过了，跳过
                    if key not in skipped_keys:
                        skipped_keys.append(key)
                    continue
                elif ".attn.output_proj.weight" in key:
                    old_key = f"layers.{layer_idx}.out_proj.weight"
                elif ".attn.output_proj.bias" in key:
                    old_key = f"layers.{layer_idx}.out_proj.bias"
                
                # 4.3 处理MLP层
                elif ".mlp.w1.weight" in key:
                    old_key = f"layers.{layer_idx}.mlp.gate_proj.weight"
                elif ".mlp.w3.weight" in key:
                    old_key = f"layers.{layer_idx}.mlp.up_proj.weight"
                elif ".mlp.w2.weight" in key:
                    old_key = f"layers.{layer_idx}.mlp.down_proj.weight"
            
            # 5. 处理最终归一化
            elif key == "transformer.norm.weight":
                old_key = "final_norm.weight"
            elif key == "transformer.norm.bias":
                old_key = "final_norm.bias"
            
            # 6. 处理位置编码
            elif key == "position_embedding.weight":
                old_key = "position_embedding.weight"
            
            # 如果找到了旧键名
            if old_key is not None:
                reverted_state_dict[old_key] = value
                reverted_count += 1
            else:
                # 其他键跳过
                if key not in skipped_keys:
                    skipped_keys.append(key)
        
        # 打印转换统计信息
        print(f"还原状态字典统计:")
        print(f"  总键数: {total_count}")
        print(f"  还原成功: {reverted_count}")
        print(f"  跳过键数: {len(skipped_keys)}")
        if skipped_keys:
            print(f"  跳过的键: {skipped_keys}")
        
        return reverted_state_dict


def test_generate_with_correct_embedding_dim():
    # 配置参数
    vocab_size = 1000
    max_length = 30
    d_model = 128
    eos_token = 999
    nhead = 4
    num_layers = 2
    dim_feedforward = 512
    
    # 必须手动创建token_embedding
    token_embedding = nn.Embedding(vocab_size, d_model)
    
    # 创建模型
    model = TokenDecoder(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        token_embedding=token_embedding,
        use_gradient_checkpointing=False,
        reduce=False
    )
    
    print("="*60)
    print("测试1：Input IDs输入 → 转为d_model维度embedding")
    input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
    generated_ids, logits = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        do_sample=False,
        temperature=1.0
    )
    print(f"输入IDs形状: {input_ids.shape}")
    print(f"生成IDs形状: {generated_ids.shape}")
    print(f"生成IDs:\n{generated_ids}")
    print(f"Logits形状: {logits.shape} (batch_size, new_tokens, vocab_size)")
    
    print("\n" + "="*60)
    print("测试2：直接输入d_model维度的embedding")
    batch_size = 2
    seq_len = 3
    input_embeddings = torch.randn(batch_size, seq_len, d_model)
    print(f"输入embedding形状: {input_embeddings.shape} → (batch_size, seq_len, d_model={d_model})")
    
    generated_ids_emb, logits_emb = model.generate(
        input_embeddings=input_embeddings,
        max_new_tokens=5,
        do_sample=True,
        temperature=0.7,
        top_k=10
    )
    print(f"生成IDs形状: {generated_ids_emb.shape}")
    print(f"生成IDs:\n{generated_ids_emb}")
    print(f"Logits形状: {logits_emb.shape}")
    
    print("\n" + "="*60)
    print("测试3：生成到EOS自动停止")
    input_ids_eos = torch.tensor([[5, 6, 7]])
    # 临时修改模型让其更容易生成EOS
    with torch.no_grad():
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            model.lm_head.weight[eos_token] *= 1000
        else:
            model.transformer.output.weight[eos_token] *= 1000
    
    generated_ids_eos, logits_eos = model.generate(
        input_ids=input_ids_eos,
        max_new_tokens=10,
        do_sample=False
    )
    print(f"输入IDs: {input_ids_eos}")
    print(f"生成IDs: {generated_ids_eos}")
    print(f"是否包含EOS: {eos_token in generated_ids_eos[0]}")
    print(f"生成长度: {len(generated_ids_eos[0])} (预期小于等于{len(input_ids_eos[0])+10})")

if __name__ == "__main__":
    test_generate_with_correct_embedding_dim()
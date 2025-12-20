import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from flash_attn import flash_attn_varlen_func, flash_attn_func
from transformers.activations import ACT2FN
import torch.utils.checkpoint as checkpoint
from typing import Protocol, Optional, Any


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


gi = 0

class EagerAttention:
    """Standard eager attention implementation, conforming to AttentionFunction protocol"""
    
    def __call__(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 is_causal: bool = False,
                 attn_dropout: float = 0.0,
                 **kwargs) -> torch.Tensor:
        """Make the class callable, delegates to forward method"""
        return self.forward(q, k, v, is_causal, attn_dropout, **kwargs)
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False,
                attn_dropout: float = 0.0,
                **kwargs) -> torch.Tensor:
        """Implements standard eager attention
        Args:
            q: Query tensor, shape: (b, s_q, n_h, h_d)
            k: Key tensor, shape: (b, s_k, n_h, h_d)
            v: Value tensor, shape: (b, s_v, n_h, h_d)
            is_causal: Whether to use causal mask (only see information before current position)
            attn_dropout: Dropout probability for attention weights
            **kwargs: Other optional parameters, such as attention mask, positional encoding, cu_seqlens, etc.
        """
        # Calculate attention scores
        h_d = q.size(-1)
        
        # Handle custom mask if provided
        mask = kwargs.get('mask', None)
        
        # Use einsum for efficient batch matrix multiplication: Q @ K^T
        # q: (b, s_q, n_h, h_d), k: (b, s_k, n_h, h_d) -> scores: (b, n_h, s_q, s_k)
        # Contract over h_d dimension, match n_h, compute s_q @ s_k
        scores = torch.einsum('bqnd, bknd -> bnqk', q, k) * \
            kwargs.get("softmax_scale", (h_d ** -0.5))
        
        # Apply custom mask (if provided)
        if mask is not None:
            # mask shape: [b, s_q, s_k] or [b, n_h, s_q, s_k]
            # scores shape: [b, n_h, s_q, s_k]
            # If mask doesn't have the head dimension, unsqueeze it
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [b, 1, s_q, s_k]
            scores = scores + mask
        
        # Apply causal mask
        if is_causal:
            assert mask is None, "Causal mask and custom mask are not supported together"
            # Flash Attention 2.1 style: right-bottom aligned causal mask
            # This supports incremental decoding where seqlen_q != seqlen_k
            s_q, s_k = q.size(1), k.size(1)
            # Show causal_mask:
            # q=2, k=2 (square, most common case):
            #   1 1
            #   1 1
            # q=2, k=4 (with common prefix):
            #   1 1 1 0
            #   1 1 1 1
            # q=4, k=2 (uncommon case, just keep same as flash attention 2.1 & sdpa):
            #   0 0
            #   0 0
            #   1 0
            #   1 1
            causal_mask = torch.tril(
                torch.ones(s_q, s_k, device=q.device), diagonal=s_k - s_q).bool()
            # Mask out positions where causal_mask is False (invert for masked_fill)
            scores.masked_fill_(~causal_mask, -float('inf'))
            
        # Calculate attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if attn_dropout > 0.0:
            # Get training mode from kwargs, default to False (eval mode) for safety
            # This matches FlashAttention's behavior where dropout is automatically handled
            training = kwargs.get('training', False)
            attn_weights = F.dropout(attn_weights, p=attn_dropout, training=training)
        
        # Calculate output
        output = torch.einsum('bnqk, bnkd -> bnqd', attn_weights, v.transpose(1, 2))
        # transpose back to (b, s_q, n_h, h_d)
        return output.transpose(1, 2)
    

class FlashAttentionDecoderLayer(nn.Module):
    """支持Flash Attention或Eager Attention的Decoder Layer（无dropout）"""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        use_flash_attn: bool = True  # 新增：控制使用哪种attention
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_flash_attn = use_flash_attn  # 保存参数
        
        # 确保head_dim是8的倍数（仅对Flash Attention有效）
        if use_flash_attn:
            assert self.head_dim * nhead == d_model, "d_model必须能被nhead整除"
            assert self.head_dim % 8 == 0, f"使用Flash Attention时head_dim必须是8的倍数，当前为{self.head_dim}"
        else:
            assert self.head_dim * nhead == d_model, "d_model必须能被nhead整除"

        # Pre-Norm层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 自注意力投影层
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.mlp = Qwen3MLP(d_model, dim_feedforward)

    def _eager_self_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int):


        """Eager Attention实现（PyTorch原生），带因果掩码"""
        batch_size = q.shape[0]
        


        if os.environ.get("debug_for_muse", "0") == "1":
            # 重塑为(batch_size, nhead, seq_len, head_dim)
            q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
            k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
            v = v.view(batch_size, seq_len, self.nhead, self.head_dim)
            return EagerAttention()(q,k,v,is_causal=True).view(batch_size, seq_len, self.d_model)

        # 重塑为(batch_size, nhead, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, nhead, seq_len, seq_len)
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=scores.dtype, device=scores.device))
        
        # 创建因果掩码（下三角矩阵）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, nhead, seq_len, head_dim)
        
        # 重塑回(batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return attn_output

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt0 = tgt

        batch_size, seq_len, _ = tgt.shape

        # 自注意力子层
        tgt_norm = self.norm1(tgt)

        # import IPython
        # IPython.embed()
        qkv = self.qkv_proj(tgt_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.use_flash_attn:
            # 重塑为(batch_size, seq_len, nheads, headdim)格式
            q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
            k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
            v = v.view(batch_size, seq_len, self.nhead, self.head_dim)

            # 使用flash_attn_func（batch版本）
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=0.0,  # 训练时可以设置dropout
                softmax_scale=1.0 / (self.head_dim ** 0.5),
                causal=True  # 启用因果掩码
            )

            # 还原原始形状: (batch_size, seq_len, d_model)
            attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        else:
            # Eager Attention模式（PyTorch原生实现）
            attn_output = self._eager_self_attention(q, k, v, seq_len)

        # 输出投影和残差连接
        attn_output = self.out_proj(attn_output)
        tgt = tgt + attn_output

        # 前馈网络子层
        tgt_norm = self.norm2(tgt)
        ffn_output = self.mlp(tgt_norm)  # 修复了这里的错误，之前引用了未定义的变量

        # 残差连接
        tgt = tgt + ffn_output

        print("fddddd")
        import IPython
        IPython.embed()
        return tgt


class PureDecoderTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        max_length: int,
        d_model: int,
        eos_token: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        token_embedding: nn.Embedding = None,
        use_flash_attn: bool = False,  # 新增：控制使用哪种attention
        use_gradient_checkpointing: bool = True,  # 新增：控制是否使用gradient checkpointing
        input_dim: int = None,
        reduce: bool = False,
        lm_head: bool = None,
        infer_id_embs_fn = None
    ):
        """简化版纯Decoder Transformer（无dropout，支持Flash Attention/Eager Attention）"""
        super().__init__()
        print(f"PureDecoderTransformer parameters: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, dim_feedforward={dim_feedforward}, use_flash_attn={use_flash_attn}, use_gradient_checkpointing={use_gradient_checkpointing}, input_dim={input_dim}, reduce={reduce}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.eos_token = eos_token
        self.use_flash_attn = use_flash_attn  # 保存参数
        self.use_gradient_checkpointing = use_gradient_checkpointing  # 保存gradient checkpointing参数
        self.input_dim = input_dim
        self.lm_head = lm_head # 一个nn.Linear层，输出logits
        self.infer_id_embs_fn = infer_id_embs_fn # 输入batchsize x seqlen x (n_q_tokens + 1) 的id, 输出batchsize x seqlen x (n_q_tokens + 1) x dim的id embedding的函数

        # 检查head_dim合法性
        head_dim = d_model // nhead
        if use_flash_attn:
            assert head_dim % 8 == 0, f"使用Flash Attention时head_dim必须是8的倍数，当前为{head_dim}"
        assert head_dim * nhead == d_model, "d_model必须能被nhead整除"

        # Token Embedding层
        if token_embedding is not None:
            self.token_embedding = token_embedding
            assert token_embedding.embedding_dim == d_model, "Embedding维度必须与d_model一致"
            assert token_embedding.num_embeddings == vocab_size, "Embedding词表大小必须与vocab_size一致"
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.position_embedding = nn.Embedding(max_length, d_model)

        # 堆叠Decoder层（传递use_flash_attn参数）
        self.layers = nn.ModuleList([
            FlashAttentionDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation="relu",
                use_flash_attn=use_flash_attn  # 传递参数
            ) for _ in range(num_layers)
        ])

        # 最终归一化
        self.final_norm = nn.LayerNorm(d_model)
        self.reduce = reduce
        if not self.reduce:
            self.input_linear = nn.Identity()
            self.output_linear = nn.Identity()
        else:
            self.input_linear = nn.Linear(self.input_dim, d_model)
            self.output_linear = nn.Linear(d_model, self.input_dim)

        # 权重初始化
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        """简单权重初始化"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.d_model ** (-0.5))
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x_emb):
        """
        前向传播（输入为embedding）
        Args:
            x_emb: (Batch, Seq_Len, d_model)
        Returns:
            out: (Batch, Seq_Len, d_model)
        """

        batch_size, seq_len, _ = x_emb.shape
        
        if not self.reduce: x_emb0 = 0
        else:  x_emb0 = x_emb

        x_emb = self.input_linear(x_emb)

        # 序列长度检查
        if seq_len > self.max_length:
            raise ValueError(f"序列长度{seq_len}超过最大限制{self.max_length}")


        # 位置编码
        positions = torch.arange(seq_len, device=x_emb.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = x_emb + pos_emb

        # print("PureDecoderTransformer.forward11111")
        # import IPython
        # IPython.embed()

        # 逐层前向传播
        if self.use_gradient_checkpointing and self.training:
            # 使用gradient checkpointing
            # def create_custom_forward(module):
            #     def custom_forward(*inputs):
            #         return module(*inputs)
            #     return custom_forward

            for layer in self.layers:
                x = checkpoint.checkpoint(
                    layer,
                    x,
                    use_reentrant=False  # 使用非重入模式，更安全
                )
        else:
            # 常规前向传播
            for layer in self.layers:
                x = layer(x)

        # 最终归一化（保持原有注释逻辑）
        # out = self.final_norm(x)
        x = self.output_linear(x)
        x = x + x_emb0
        x_before_lm_head = x
        if self.lm_head is not None:
            x = self.lm_head(x)
        # import IPython
        # IPython.embed()
        return x

    def forward_with_tokens(self, tokens: torch.Tensor):
        """
        完善的辅助方法：支持输入token ID直接前向传播，可选择使用infer_id_embs_fn
        Args:
            tokens: (Batch, Seq_Len) token ID序列
        Returns:
            out: (Batch, Seq_Len, d_model) 或 (Batch, Seq_Len, vocab_size) 如果lm_head不为None
        """
        # 如果提供了infer_id_embs_fn，则使用它来获取token embeddings
        if self.infer_id_embs_fn is not None:
            # 假设infer_id_embs_fn可以处理形状为(batch_size, seq_len)的输入
            # 并返回形状为(batch_size, seq_len, d_model)的embeddings
            x_emb = self.infer_id_embs_fn(tokens)
        else:
            # 否则使用标准的token embedding
            x_emb = self.token_embedding(tokens)
        
        # 通过主forward方法传递
        output = self.forward(x_emb)

        return output
    

    def generate(self, input_ids: torch.Tensor = None, input_embeddings: torch.Tensor = None, 
                 max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = None, top_p: float = None,
                 do_sample: bool = False, pad_token_id: int = None,
                 attention_mask: torch.Tensor = None, return_logits: bool = True,
                 only_last=False
                 ):
        """
        生成函数：统一使用input_embeddings作为输入（prefill后也保持embedding输入）
        """
        self.eval()  # 设置为评估模式
        # print(f"input_embeddings={input_embeddings.shape}")
        # 检查输入合法性
        if (input_ids is None and input_embeddings is None) or \
           (input_ids is not None and input_embeddings is not None):
            raise ValueError("必须提供input_ids或input_embeddings中的一个，且不能同时提供")
        
        # print(f"tokenhead\ninput_embeddings={input_embeddings[...,:4]}")
        global gi
        gi += 1
        with torch.no_grad():
            # 1. 统一初始输入为embeddings（无论输入是ids还是embeddings）
            if input_ids is not None:
                batch_size, seq_len = input_ids.shape
                device = input_ids.device
                generated_ids = input_ids.clone()
                
                # 从input_ids生成初始embeddings（维度保证为d_model）
                if self.infer_id_embs_fn is not None:
                    current_embeddings = self.infer_id_embs_fn(input_ids, group_size=1)
                else:
                    current_embeddings = self.token_embedding(input_ids)
                
                # 验证embedding维度
                # assert current_embeddings.shape[-1] == self.d_model, f"初始embedding维度必须为{self.d_model}"
            
            else:  # input_embeddings is not None
                batch_size, seq_len, emb_dim = input_embeddings.shape
                device = input_embeddings.device
                
                # 验证输入embedding维度
                # assert emb_dim == self.d_model, f"输入embedding维度必须为{self.d_model}，当前为{emb_dim}"
                
                current_embeddings = input_embeddings.clone()
                # 初始化generated_ids为占位符（无实际token时用0填充）
                generated_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
            
            # 初始化logits存储
            logits_list = []
            # 跟踪是否生成EOS
            # finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            # 2. 自回归生成循环（始终用embeddings输入）
            for _ in range(max_new_tokens):
                # 序列长度限制检查
                if current_embeddings.shape[1] >= self.max_length:
                    break
                
                # print(f"current_embeddings.shape", current_embeddings.shape)
                # 前向传播（仅用embeddings输入）
                logits = self.forward(current_embeddings)

                # print(f"logits.shape", logits.shape)
                next_token_logits = logits[:, -1, :]
                
                # print(f"next_token_logits.shape", next_token_logits.shape)
                # print(f"next_token_logits.argmax(-1)", next_token_logits.argmax(-1))
                # topk(logits, k=K, dim=-1).indices
                # print(f"next_token_logits.top6(-1)", next_token_logits.topk(6, dim=-1).indices)
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
                
                # print(f"333333", next_token_logits.shape)

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
                
                # print(f"next_token_logits={next_token_logits.shape}")
                # 生成下一个token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                # print(f"next_token={next_token}")
                # 更新生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                # print(f"next_token={next_token}")
                # 生成新token的embedding并更新输入embeddings（关键：保持embedding输入）
                if self.infer_id_embs_fn is not None:
                    next_token_emb = self.infer_id_embs_fn(next_token, group_size=1)[:,:,0] # group size=1
                else:
                    next_token_emb = self.token_embedding(next_token)
                
                # print(f"current_embeddingscurrent_embeddings", current_embeddings.shape, next_token_emb.shape)
                current_embeddings = torch.cat([current_embeddings, next_token_emb], dim=1)
                

                # print(f"next_token_emb={next_token_emb}")
                # print(f"33343333", next_token_emb.shape, next_token.shape, self.eos_token)

                # 检查EOS停止条件
                if self.eos_token is not None:
                    if next_token[-1,0] == self.eos_token:
                        break
                    # finished = finished | (next_token.squeeze(-1) == self.eos_token)
                    # # print(f"finished={finished}")
                    # if finished.all():
                    #     break

            # print(f"generated_ids={generated_ids.shape}")
            # if 0:
            #     from recovlm.utils.ds_utils import print_input_info
            #     print_input_info({
            #             "current_embeddings": current_embeddings,
            #             "logits_list": torch.stack(logits_list, dim=1),
            #             "generated_ids": generated_ids,
            #         }, "keye_token_embedding", save_path=f"tokenizer_debug_{gi}.pth")
            #     else:
            #         print(f"generated_ids={generated_ids.shape}")
            # 处理返回结果
            if only_last:
                generated_ids = generated_ids[..., -1:,:]
                logits_list = logits_list[-1:]
            # print("\n\n")
            if return_logits and logits_list:
                logits_tensor = torch.stack(logits_list, dim=1)
                # print(f"logits_tensor={logits_tensor.shape}, generated_ids={generated_ids.shape}, generated_ids={generated_ids}")

                # 补全logits长度（如果提前停止）
                # if logits_tensor.size(1) < max_new_tokens:
                #     pad_shape = (batch_size, max_new_tokens - logits_tensor.size(1), self.vocab_size)
                #     pad_tensor = torch.zeros(pad_shape, dtype=logits_tensor.dtype, device=device)
                #     logits_tensor = torch.cat([logits_tensor, pad_tensor], dim=1)
                return generated_ids, logits_tensor
            

            return generated_ids

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast  # 混合精度训练工具

# 导入你定义的模型（确保原模型代码已正确定义）

def test_backward_pass_bf16():
    """测试bfloat16精度下的反向传播有效性（CUDA优先）"""
    # ====================== 1. 基础配置 ======================
    vocab_size = 100  # 词表大小
    d_model = 128     # 模型维度（128//8=16，满足head_dim是8的倍数）
    nhead = 8         # 注意力头数
    num_layers = 2    # Decoder层数
    max_length = 50   # 最大序列长度
    eos_token = 0     # EOS标记
    batch_size = 4    # 批次大小（bfloat16可适当增大，这里保持适中）
    epochs = 3        # 训练轮数
    lr = 1e-4         # 学习率（bfloat16数值稳定性好，可保持原学习率）

    # ====================== 2. 设备和精度检查 ======================
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境无CUDA支持，无法使用bfloat16精度！")
    
    device = torch.device("cuda")
    # 检查GPU是否支持bfloat16（T4、A10、A100、H100等均支持）
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前GPU不支持bfloat16精度，请更换支持的GPU（如T4/A10/A100）！")
    
    print(f"使用设备: {device}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"精度模式: bfloat16")

    # ====================== 3. 生成bfloat16格式的模拟数据 ======================
    seq_len = 20  # 随机序列长度
    # 生成CPU上的整数序列，再移到CUDA
    train_data = torch.randint(
        low=1, high=vocab_size, 
        size=(100, seq_len),  # 100个样本
        dtype=torch.long  # token序列必须是long类型
    ).to(device)

    # 构建数据集和数据加载器（数据已在CUDA上）
    dataset = TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    # pin_memory=False（数据已在CUDA，无需.pin_memory）

    # ====================== 4. 初始化模型（bfloat16精度） ======================
    model = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=512
    ).to(device, dtype=torch.bfloat16)  # 模型参数直接设为bfloat16

    # 输出投影层（同样使用bfloat16）
    output_proj = nn.Linear(d_model, vocab_size).to(device, dtype=torch.bfloat16)

    # ====================== 5. 优化器和损失函数 ======================
    # 优化器：bfloat16下建议使用AdamW，无需特殊调整
    optimizer = optim.AdamW(
        list(model.parameters()) + list(output_proj.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 损失函数：CrossEntropyLoss支持bfloat16输入（自动处理精度转换）
    criterion = nn.CrossEntropyLoss(ignore_index=eos_token).to(device)

    # ====================== 6. 训练循环（bfloat16混合精度） ======================
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0]  # (batch_size, seq_len) -> long类型（CUDA上）
            batch_size, seq_len = x.shape

            # 构建输入和目标（自回归预测下一个token）
            input_seq = x[:, :-1]  # (batch_size, seq_len-1)
            target_seq = x[:, 1:]  # (batch_size, seq_len-1)

            # ====================== 前向传播（bfloat16） ======================
            with autocast(dtype=torch.bfloat16):  # 启用bfloat16自动转换
                # 1. Token Embedding（输入是long，embedding输出自动转为bfloat16）
                token_emb = model.token_embedding(input_seq)  # (batch_size, seq_len-1, d_model) -> bfloat16
                
                # 2. 模型前向传播（所有中间计算均为bfloat16）
                model_output = model(token_emb)  # (batch_size, seq_len-1, d_model) -> bfloat16
                
                # 3. 输出投影（bfloat16 -> bfloat16）
                logits = output_proj(model_output)  # (batch_size, seq_len-1, vocab_size) -> bfloat16

                # 4. 计算损失（CrossEntropyLoss会自动处理bfloat16输入）
                loss = criterion(
                    logits.reshape(-1, vocab_size),  # (batch_size*(seq_len-1), vocab_size)
                    target_seq.reshape(-1)           # (batch_size*(seq_len-1),)
                )

            # ====================== 反向传播和参数更新 ======================
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # bfloat16下梯度计算正常（PyTorch自动处理精度）
            optimizer.step()       # 更新参数（参数保持bfloat16）

            # 累加损失（将bfloat16损失转为float32避免精度丢失）
            total_loss += loss.float().item() * batch_size

        # 计算平均损失
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

        # ====================== 验证bfloat16下的反向传播有效性 ======================
        # 检查模型参数梯度（bfloat16梯度是否有效）
        grad_available = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 检查梯度是否非零（bfloat16下允许微小数值误差）
                if not torch.allclose(param.grad, torch.zeros_like(param.grad, dtype=torch.bfloat16), atol=1e-5):
                    grad_norm = param.grad.norm().item()
                    print(f"✓ 参数 {name} 存在有效梯度（bfloat16，梯度范数：{grad_norm:.4f}）")
                    grad_available = True
                    break
        
        if not grad_available:
            print("✗ 模型参数无有效梯度！反向传播可能失败！")
        else:
            # 检查输出投影层梯度
            for name, param in output_proj.named_parameters():
                if param.grad is not None and not torch.allclose(param.grad, torch.zeros_like(param.grad, dtype=torch.bfloat16), atol=1e-5):
                    print(f"✓ 输出投影层 {name} 存在有效梯度（bfloat16，梯度范数：{param.grad.norm().item():.4f}）")
                    break

        # 修复：使用detach()从计算图分离张量后再转numpy
        first_param = next(model.parameters())
        print(f"✓ 模型参数精度: {first_param.dtype}, 部分数值: {first_param[:2, :2].float().detach().cpu().numpy()}")

    print("\n" + "="*60)
    print("bfloat16精度训练完成！反向传播有效性验证通过（CUDA+bfloat16）")
    print("="*60)


def test_eager():
    # 配置参数
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_layers = 3
    max_length = 128
    eos_token = 1

    # 1. 使用Flash Attention（默认）
    flash_model = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        use_flash_attn=True
    ).cuda().bfloat16()

    # 2. 使用Eager Attention（PyTorch原生）
    # eager_model = PureDecoderTransformer(
    #     vocab_size=vocab_size,
    #     max_length=max_length,
    #     d_model=d_model,
    #     eos_token=eos_token,
    #     nhead=nhead,
    #     num_layers=num_layers,
    #     use_flash_attn=False
    # ).cuda().bfloat16()

    # 测试输入
    batch_size = 2
    seq_len = 32
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()  # 随机token ID
    x_emb = flash_model.token_embedding(tokens)  # 生成embedding

    # 前向传播
    flash_output = flash_model(x_emb.cuda().bfloat16())


    eager_model = flash_model
    eager_model.use_flash_attn = False
    for sub_model in eager_model.layers:
        sub_model.use_flash_attn = False

    eager_output = eager_model(x_emb.cuda().bfloat16())

    # 也可以直接输入token ID
    flash_output2 = flash_model.forward_with_tokens(tokens.cuda())

    print(f"Flash Attention输出形状: {flash_output.shape}")
    print(f"Eager Attention输出形状: {eager_output.shape}")
    print(f"直接输入token的输出形状: {flash_output2.shape}")
    print(flash_output)
    print(eager_output)


def test_gradient_checkpointing():
    """测试gradient checkpointing的有效性"""
    # ====================== 1. 增大模型配置 ======================
    vocab_size = 10000
    d_model = 1024  # 增大模型维度
    nhead = 16
    num_layers = 12  # 增加层数
    max_length = 512
    eos_token = 0
    batch_size = 16  # 增大batch size
    seq_len = 128    # 增长序列长度

    # ====================== 2. 设备检查 ======================
    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA环境测试显存节省效果")
    
    device = torch.device("cuda")
    print(f"使用设备: {device}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"模型配置: d_model={d_model}, layers={num_layers}, batch_size={batch_size}, seq_len={seq_len}")

    # ====================== 3. 生成测试数据 ======================
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # ====================== 4. 创建模型（不使用Flash Attention） ======================
    # 不使用gradient checkpointing的模型
    model_no_checkpoint = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        use_flash_attn=False,  # 使用标准attention更能体现效果
        use_gradient_checkpointing=False,
        dim_feedforward=4096
    ).to(device)

    # 使用gradient checkpointing的模型
    model_with_checkpoint = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        use_flash_attn=False,
        use_gradient_checkpointing=True,
        dim_feedforward=4096
    ).to(device)

    # 确保参数相同
    model_with_checkpoint.load_state_dict(model_no_checkpoint.state_dict())

    # 设置为训练模式
    model_no_checkpoint.train()
    model_with_checkpoint.train()

    # ====================== 5. 精确测量内存使用 ======================
    def measure_memory_usage(model, inputs):
        """精确测量模型的内存使用"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 预热
        with torch.no_grad():
            for _ in range(2):
                model.forward_with_tokens(inputs)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 前向+反向传播
        output = model.forward_with_tokens(inputs)
        loss = output.sum()
        
        forward_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        loss.backward()
        
        total_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        return forward_mem, total_mem

    # ====================== 6. 测试不使用checkpointing的模型 ======================
    print("\n测试不使用gradient checkpointing的模型：")
    forward_mem_no_ckpt, total_mem_no_ckpt = measure_memory_usage(model_no_checkpoint, tokens)
    print(f"前向传播内存: {forward_mem_no_ckpt:.2f} MB")
    print(f"总内存使用: {total_mem_no_ckpt:.2f} MB")
    
    # 重置梯度
    model_no_checkpoint.zero_grad()

    # ====================== 7. 测试使用checkpointing的模型 ======================
    print("\n测试使用gradient checkpointing的模型：")
    forward_mem_ckpt, total_mem_ckpt = measure_memory_usage(model_with_checkpoint, tokens)
    print(f"前向传播内存: {forward_mem_ckpt:.2f} MB")
    print(f"总内存使用: {total_mem_ckpt:.2f} MB")

    # ====================== 8. 结果比较 ======================
    print("\n" + "="*60)
    print("结果比较：")
    print(f"不使用checkpointing的总内存: {total_mem_no_ckpt:.2f} MB")
    print(f"使用checkpointing的总内存: {total_mem_ckpt:.2f} MB")
    
    mem_saving = total_mem_no_ckpt - total_mem_ckpt
    mem_saving_pct = (mem_saving / total_mem_no_ckpt) * 100
    
    if mem_saving > 0:
        print(f"✓ 内存节省: {mem_saving:.2f} MB ({mem_saving_pct:.2f}%)")
    else:
        print(f"✗ 内存增加: {abs(mem_saving):.2f} MB ({abs(mem_saving_pct):.2f}%)")
    
    # 验证输出一致性
    with torch.no_grad():
        output_no_ckpt = model_no_checkpoint.forward_with_tokens(tokens)
        output_ckpt = model_with_checkpoint.forward_with_tokens(tokens)
    
    output_diff = torch.norm(output_no_ckpt - output_ckpt).item()
    print(f"输出差异: {output_diff:.6f}")
    
    if output_diff < 1e-4:
        print("✓ 计算结果保持一致")
    else:
        print("✗ 计算结果不一致")
    
    print("="*60)
    
    # 额外测试：不同batch size下的表现
    print("\n不同batch size下的显存使用对比：")
    for bs in [8, 16, 32]:
        if bs * seq_len * d_model > 1e9:  # 防止OOM
            continue
            
        test_tokens = torch.randint(0, vocab_size, (bs, seq_len), device=device)
        
        # 无checkpointing
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = model_no_checkpoint.forward_with_tokens(test_tokens)
        (out.sum()).backward()
        mem_no_ckpt = torch.cuda.max_memory_allocated() / 1024**2
        
        # 有checkpointing
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        out = model_with_checkpoint.forward_with_tokens(test_tokens)
        (out.sum()).backward()
        mem_ckpt = torch.cuda.max_memory_allocated() / 1024**2
        
        saving = mem_no_ckpt - mem_ckpt
        saving_pct = (saving / mem_no_ckpt) * 100
        
        print(f"Batch size {bs}:")
        print(f"  无checkpointing: {mem_no_ckpt:.2f} MB")
        print(f"  有checkpointing: {mem_ckpt:.2f} MB")
        print(f"  显存变化: {saving:+.2f} MB ({saving_pct:+.2f}%)")



def generate_demo():
    """
    演示如何使用PureDecoderTransformer的generate函数
    """
    import time
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型配置
    vocab_size = 10
    max_length = 512
    d_model = 256
    nhead = 8
    num_layers = 4
    eos_token = 1
    
    # 确保d_model能被nhead整除
    assert d_model % nhead == 0, "d_model必须能被nhead整除"
    
    # 创建lm_head层
    lm_head = nn.Linear(d_model, vocab_size, bias=False).to(device).bfloat16()
    embedding = nn.Embedding(vocab_size, d_model).to(device).bfloat16()

    # 定义一个简单的infer_id_embs_fn示例
    def infer_id_embs_fn(ids):
        # 这里我们简单地使用标准embedding，但在实际应用中可能有更复杂的逻辑
        # 例如，如果ids包含特殊格式或需要特殊处理
        return embedding(ids)
    
    # 创建模型
    model = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        lm_head=lm_head,
        infer_id_embs_fn=infer_id_embs_fn,
        use_flash_attn=torch.cuda.is_available()
    ).to(device).bfloat16()
    
    # 生成一些随机输入
    batch_size = 2
    seq_len = 5
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输入内容: {input_ids}")
    
    # 测试不同的生成配置
    print("\n测试贪婪解码:")
    start_time = time.time()
    greedy_output = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False
    )
    end_time = time.time()
    print(f"贪婪解码输出形状: {greedy_output[0].shape} {greedy_output[1].shape}")
    print(f"贪婪解码输出内容: {greedy_output}")
    print(f"贪婪解码用时: {end_time - start_time:.4f}秒")
    
    print("\n测试采样解码:")
    start_time = time.time()
    sample_output = model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        return_logits=True
    )
    end_time = time.time()
    print(f"采样解码输出形状: {sample_output[0].shape} {sample_output[1].shape}")
    print(f"采样解码输出内容: {sample_output}")
    print(f"采样解码用时: {end_time - start_time:.4f}秒")
    
    # 测试forward_with_tokens方法
    print("\n测试forward_with_tokens方法:")
    with torch.no_grad():
        output = model.forward_with_tokens(input_ids)
        print(f"forward_with_tokens输出形状: {output.shape}")
    
    print("\n演示完成！")



def test_generate_with_correct_embedding_dim():
    # 配置参数
    vocab_size = 1000
    max_length = 30
    d_model = 128  # embedding维度必须等于d_model
    eos_token = 999  # EOS token ID
    nhead = 4
    num_layers = 2
    dim_feedforward = 512
    
    # 必须手动创建token_embedding（维度为d_model）
    token_embedding = nn.Embedding(vocab_size, d_model)  # 确保embedding_dim=d_model
    # 创建LM Head（输入d_model，输出vocab_size）
    lm_head = nn.Linear(d_model, vocab_size)
    
    # 初始化模型（禁用Flash Attention便于测试）
    model = PureDecoderTransformer(
        vocab_size=vocab_size,
        max_length=max_length,
        d_model=d_model,
        eos_token=eos_token,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        token_embedding=token_embedding,  # 必须传入
        use_flash_attn=False,
        use_gradient_checkpointing=False,
        lm_head=lm_head,
        reduce=False  # 确保不修改embedding维度
    )
    
    print("="*60)
    print("测试1：Input IDs输入 → 转为d_model维度embedding")
    input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])  # (2, 3)
    generated_ids, logits = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        do_sample=False,  # 贪婪解码
        temperature=1.0
    )
    print(f"输入IDs形状: {input_ids.shape}")
    print(f"生成IDs形状: {generated_ids.shape}")
    print(f"生成IDs:\n{generated_ids}")
    print(f"Logits形状: {logits.shape} (batch_size, new_tokens, vocab_size)")
    
    print("\n" + "="*60)
    print("测试2：直接输入d_model维度的embedding")
    # 手动创建维度为d_model的input_embeddings
    batch_size = 2
    seq_len = 3
    input_embeddings = torch.randn(batch_size, seq_len, d_model)  # 确保最后一维是d_model
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
    print("测试3：验证embedding维度检查")
    try:
        # 尝试输入错误维度的embedding（应该报错）
        wrong_dim_embedding = torch.randn(2, 3, 64)  # 维度不是d_model=128
        model.generate(input_embeddings=wrong_dim_embedding)
    except AssertionError as e:
        print(f"正确捕获维度错误：{e}")
    
    print("\n" + "="*60)
    print("测试4：生成到EOS自动停止")
    # 构造测试输入
    input_ids_eos = torch.tensor([[5, 6, 7]])
    # 临时修改模型让其更容易生成EOS
    with torch.no_grad():
        model.lm_head.weight[eos_token] *= 1000  # 大幅提高EOS的logits
    
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



'''
请你阅读muse/models/keye_ar/token_decoder_ori.py的代码
然后在muse/models/keye_ar/token_decoder.py这个脚本中重构一个相同的代码，要求新的代码不在依赖于transformers包，
并且尽量复用muse/layers/中的类，尽可能减少代码量，代码风格参考muse/models/qwen3/modeling.py。对于新的代码，你也需要实现一个test_generate_with_correct_embedding_dim函数，测试生成效果。
若有必要自己实现一些子模块，可以继承原来的模块。


请你测试muse/models/keye_ar/token_decoder.py中的convert_hf_state_dict函数，要求：使用相同的config初始化一个token_decoder_ori和token_decoder。然后把token_decoder_ori的statedict转成token_decoder的。并且测试两个模型的前向，预期相同。
'''
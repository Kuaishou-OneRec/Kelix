import torch
import torch.nn.functional as F

from einops import rearrange

def select_representative_embeddings_for_codebook(llm_embeddings, codebook_size, normalize=True):
    """
    从LLM embeddings中选择最具代表性的embedding用于codebook初始化
    基于平均距离的top-k选择方法
    """
    vocab_size, dim = llm_embeddings.shape
    device = llm_embeddings.device
    
    if normalize:
        llm_embeddings = F.normalize(llm_embeddings, p=2, dim=-1)
    
    print(f"从 {vocab_size:,} 个embedding中选择 {codebook_size:,} 个最具代表性的...")
    
    # 使用余弦距离（适合大词汇表如Qwen的150K）
    # 计算余弦相似度矩阵
    similarity_matrix = torch.mm(llm_embeddings, llm_embeddings.t())
    
    # 余弦距离 = 1 - 余弦相似度
    distance_matrix = 1 - similarity_matrix
    
    # 排除对角线（自身距离）
    mask = torch.eye(vocab_size, device=device).bool()
    distance_matrix = distance_matrix.masked_fill(mask, 0)
    
    # 计算每个embedding的平均距离
    avg_distances = distance_matrix.sum(dim=1) / (vocab_size - 1)
    
    # 选择平均距离最大的top-k个
    _, selected_indices = torch.topk(avg_distances, codebook_size, largest=True)
    selected_embeddings = llm_embeddings[selected_indices]
    
    # print(f"选择完成！平均距离范围: [{avg_distances[selected_indices].min().item():.4f}, {avg_distances[selected_indices].max().item():.4f}]")    
    
    return selected_embeddings, selected_indices
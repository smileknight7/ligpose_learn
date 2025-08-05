import torch
import math

#time_embedding 过程中要考虑频率偏好的问题

def sinusoidal_embedding_v1(x, dim, device):
    """版本1：分母是 half_dim"""
    half_dim = dim // 2
    emb_scale = math.log(10000) / half_dim
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb_scale)
    emb = x[:, None] * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)

def sinusoidal_embedding_v2(x, dim, device):
    """版本2：分母是 half_dim - 1"""
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    emb = x[:, None] * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 16
    # 测试输入时间步，batch_size=4，值为0~3
    t = torch.arange(4, dtype=torch.float32, device=device)
    
    emb1 = sinusoidal_embedding_v1(t, dim, device)
    emb2 = sinusoidal_embedding_v2(t, dim, device)
    
    print("Embedding version 1:\n", emb1)
    print("\nEmbedding version 2:\n", emb2)
    
    diff = (emb1 - emb2).abs()
    print("\nMax absolute difference between embeddings:", diff.max().item())

if __name__ == "__main__":
    main()
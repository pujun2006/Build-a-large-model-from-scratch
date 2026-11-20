import torch
import torch.nn as nn
import math

# 自注意力机制实现（包含单向掩码）
class GPT2SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(GPT2SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size需为heads的整数倍"

        # 定义线性层用于生成Q、K、V矩阵
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 生成多头矩阵
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = query.view(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算缩放点积注意力
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)

        # 使用单向掩码
        mask = torch.tril(torch.ones(query_len, key_len)).expand(N, self.heads, query_len, key_len)
        energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 计算注意力权重并应用于值矩阵
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)

# 前馈网络的实现
class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Transformer块的实现，包含GPT-2中的注意力机制和前馈网络
class GPT2Block(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super(GPT2Block, self).__init__()
        self.attention = GPT2SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x, mask=None)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return self.dropout(out)

# 参数设置
embed_size = 128
heads = 8
hidden_dim = 512
dropout = 0.1
seq_length = 10
batch_size = 2

# 初始化GPT-2块
gpt2_block = GPT2Block(embed_size, heads, hidden_dim, dropout)

# 随机生成输入数据
input_data = torch.rand(batch_size, seq_length, embed_size)

# 前向传播
output = gpt2_block(input_data)
print("GPT-2注意力机制输出形状:", output.shape)
print("GPT-2注意力机制输出:\n", output)


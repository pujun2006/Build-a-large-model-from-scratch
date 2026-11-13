import torch
import torch.nn as nn
import math


# 自注意力机制的实现
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size需要是heads的整数倍"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 分头计算Q, K, V矩阵
        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = query.view(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

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


# Transformer层堆叠的实现
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return self.dropout(out)


# GPT-2模型的层堆叠部分
class GPT2LayerStack(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, num_layers, dropout):
        super(GPT2LayerStack, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# 模拟输入
embed_size = 128
heads = 8
hidden_dim = 512
num_layers = 4
dropout = 0.1
seq_length = 10
batch_size = 2

# 初始化GPT-2层堆叠模型
gpt2_layer_stack = GPT2LayerStack(embed_size, heads, hidden_dim, num_layers, dropout)

# 随机生成输入数据和掩码
input_data = torch.rand(batch_size, seq_length, embed_size)
mask = None  # GPT-2通常不需要掩码，因为生成过程是单向的

# 前向传播
output = gpt2_layer_stack(input_data, mask)
print("层堆叠模型输出形状:", output.shape)
print("层堆叠模型输出:\n", output)
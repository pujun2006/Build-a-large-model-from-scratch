import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置随机种子
torch.manual_seed(42)

# 自注意力机制的实现
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size需要是heads的整数倍"

        # 线性变换用于生成Q、K、V矩阵
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

        # 计算Q与K的点积除以缩放因子
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 计算注意力权重
        attention = torch.softmax(energy, dim=-1)

        # 注意力权重乘以V
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)

# 设置参数
embed_size = 128  # 嵌入维度
heads = 8  # 多头数量
seq_length = 10  # 序列长度
batch_size = 2  # 批大小

# 创建随机输入
values = torch.rand((batch_size, seq_length, embed_size))
keys = torch.rand((batch_size, seq_length, embed_size))
queries = torch.rand((batch_size, seq_length, embed_size))

# 初始化自注意力层
self_attention_layer = SelfAttention(embed_size, heads)

# 前向传播
output = self_attention_layer(values, keys, queries, mask=None)

print("输出的形状:", output.shape)
print("自注意力机制的输出:\n", output)

# 进一步展示注意力权重计算
class SelfAttentionWithWeights(SelfAttention):
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

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

        return self.fc_out(out), attention

# 使用带有权重输出的自注意力层
self_attention_layer_with_weights = SelfAttentionWithWeights(embed_size, heads)
output, attention_weights = self_attention_layer_with_weights(values, keys, queries, mask=None)

print("注意力权重形状:", attention_weights.shape)
print("注意力权重:\n", attention_weights)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 修正后的多头自注意力机制 (Multi-Head Self Attention)
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size 必须是 heads 的整数倍"

        # 标准做法：投影矩阵映射整个 embed_size (例如 768 -> 768)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 1. 线性变换，形状变为 [N, Seq_Len, embed_size]
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # 2. 切分多头并转换维度形状 -> [N, heads, Seq_Len, head_dim]
        values = values.view(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        queries = queries.view(N, query_len, self.heads, self.head_dim).transpose(1, 2)

        # 3. 缩放点积注意力计算
        # queries: [N, heads, query_len, head_dim]
        # keys.transpose(-2, -1): [N, heads, head_dim, key_len]
        # energy 形状: [N, heads, query_len, key_len]
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 4. Mask 掩码处理
        if mask is not None:
            # 确保 mask 维度为 [N, 1, query_len, key_len] 以适配多头
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        # 5. 注意力权重乘以 Value: [N, heads, query_len, head_dim]
        out = torch.matmul(attention, values)

        # 6. 拼接多个头的输出 -> [N, query_len, embed_size]
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.heads * self.head_dim)

        return self.fc_out(out)


# 2. 定义前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))  # BERT 官方默认使用 GELU 激活函数


# 3. Transformer 编码器块 (标准 Post-LN 架构)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        # 修正残差与 LayerNorm 顺序
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        return out


# 4. BERT 编码器堆叠
class BERTEncoder(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, num_layers, dropout):
        super(BERTEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# 模拟运行
embed_size = 768
heads = 12
forward_expansion = 4
num_layers = 12
dropout = 0.1
seq_length = 20
batch_size = 2

bert_encoder = BERTEncoder(embed_size, heads, forward_expansion, num_layers, dropout)

input_data = torch.rand(batch_size, seq_length, embed_size)
mask = torch.ones(batch_size, seq_length, seq_length)

output = bert_encoder(input_data, mask)
print("编码器堆叠输出形状:", output.shape)
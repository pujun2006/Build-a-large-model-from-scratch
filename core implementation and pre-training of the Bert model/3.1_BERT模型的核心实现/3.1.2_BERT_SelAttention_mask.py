import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# 1. 修正后的标准多头自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size必须是heads的整数倍"

        # 标准做法：基于全局 embed_size 进行投影
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 转换为多头维度: [N, Seq_Len, Heads, Head_Dim] -> [N, Heads, Seq_Len, Head_Dim]
        values = self.values(values).view(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim).transpose(1, 2)

        # 矩阵乘法计算 Attention Energy
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        # 加权拼接多头输出
        out = torch.matmul(attention, values)  # [N, Heads, Query_Len, Head_Dim]
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)

        return self.fc_out(out)


# 2. 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# 3. Transformer编码器块
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
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        return out


# 4. 完整的 BERT 模型 (增加了 Embedding 层与 MLM 预测头)
class BERTModel(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, forward_expansion, num_layers, dropout, max_len=512):
        super(BERTModel, self).__init__()
        # 修正：添加 Token & Position Embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # 修正：添加 MLM 输出预测头 (映射回 vocab_size)
        self.mlm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device)

        # 向量嵌入融合
        out = self.dropout(self.token_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, mask)

        # 预测掩码位置的 Token
        logits = self.mlm_head(out)
        return out, logits


# 5. 掩码数据集创建函数
def create_masked_data(inputs, mask_token_id, vocab_size, mask_prob=0.15):
    inputs_with_masks = inputs.clone()
    labels = inputs.clone()

    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            prob = random.random()
            if prob < mask_prob:
                prob /= mask_prob
                if prob < 0.8:
                    inputs_with_masks[i, j] = mask_token_id  # 80% 替换为 [MASK]
                elif prob < 0.9:
                    inputs_with_masks[i, j] = random.randint(0, vocab_size - 1)  # 10% 随机词
                # 10% 保持不变
            else:
                labels[i, j] = -100  # -100 是 PyTorch 计算交叉熵时默认忽略的 index

    return inputs_with_masks, labels


# 超参数定义
vocab_size = 30522
embed_size = 768
num_layers = 12
heads = 12
forward_expansion = 4
dropout = 0.1
seq_length = 20
batch_size = 2
mask_token_id = 103

# 初始化模型
model = BERTModel(vocab_size, embed_size, heads, forward_expansion, num_layers, dropout)

# 生成模拟输入数据
input_data = torch.randint(0, vocab_size, (batch_size, seq_length))
mask = torch.ones(batch_size, seq_length, seq_length)

# 生成掩码任务数据
masked_data, labels = create_masked_data(input_data, mask_token_id, vocab_size)

# 前向传播
encoder_out, logits = model(masked_data, mask)

# 计算 MLM 任务的 Loss 示例
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

print("编码器隐藏层特征形状:", encoder_out.shape)  # [2, 20, 768]
print("预测 Logits 形状:", logits.shape)  # [2, 20, 30522]
print("掩码预测 CrossEntropy Loss:", loss.item())

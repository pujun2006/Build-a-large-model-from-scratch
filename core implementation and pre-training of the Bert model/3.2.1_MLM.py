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

        # 标准做法：在全局 embed_size (768) 上定义投影
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 映射并重塑形状: [N, Seq_Len, Heads, Head_Dim] -> 转置为 [N, Heads, Seq_Len, Head_Dim]
        values = self.values(values).view(N, value_len, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim).transpose(1, 2)
        queries = self.queries(query).view(N, query_len, self.heads, self.head_dim).transpose(1, 2)

        # 计算 Attention 能量: [N, Heads, Q_len, K_len]
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 扩展 Heads 维度
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        # 加权求和并复原形状
        out = torch.matmul(attention, values)  # [N, Heads, Query_len, Head_Dim]
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)

        return self.fc_out(out)


# 2. 前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# 3. 修正残差连接顺序的 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 标准 Post-LN 结构：x + Dropout(SubLayer(x)) -> LayerNorm
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        return out


# 4. BERT 编码器
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


# 5. 创建掩码数据函数 (保持原逻辑，注意忽略未掩码位置 -100)
def create_masked_data(inputs, mask_token_id, vocab_size, mask_prob=0.15):
    inputs_with_masks = inputs.clone()
    labels = inputs.clone()

    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            prob = random.random()
            if prob < mask_prob:
                prob /= mask_prob
                if prob < 0.8:
                    inputs_with_masks[i, j] = mask_token_id  # 80% 概率替换为 [MASK]
                elif prob < 0.9:
                    inputs_with_masks[i, j] = random.randint(0, vocab_size - 1)  # 10% 概率替换为随机词
                # 10% 保持不变
            else:
                labels[i, j] = -100  # 忽略非掩码位置

    return inputs_with_masks, labels


# 6. 完整的 MLM 任务模型 (补充了 Position Embedding)
class MLMTask(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, forward_expansion, num_layers, dropout, max_len=512):
        super(MLMTask, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)  # 修正：补充位置嵌入
        self.dropout = nn.Dropout(dropout)
        self.bert_encoder = BERTEncoder(embed_size, heads, forward_expansion, num_layers, dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device)

        # 词嵌入 + 位置嵌入融合
        embeddings = self.dropout(self.token_embedding(x) + self.position_embedding(positions))
        encoder_out = self.bert_encoder(embeddings, mask)
        logits = self.fc_out(encoder_out)
        return logits


# --- 模拟运行与测试 ---
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
mlm_model = MLMTask(vocab_size, embed_size, heads, forward_expansion, num_layers, dropout)

# 生成模拟输入
input_data = torch.randint(0, vocab_size, (batch_size, seq_length))
mask = torch.ones(batch_size, seq_length, seq_length)

# 构造 Mask 数据
masked_data, labels = create_masked_data(input_data, mask_token_id, vocab_size)

# 前向传播与损失计算
output_logits = mlm_model(masked_data, mask)
loss_fct = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略 -100 的标签
masked_loss = loss_fct(output_logits.view(-1, vocab_size), labels.view(-1))

print("运行成功！")
print("MLM任务损失:", masked_loss.item())
print("模型输出形状:", output_logits.shape)  # 期望形状: [2, 20, 30522]
import torch
import torch.nn as nn
import math
import random


# 位置编码器的实现，生成位置编码矩阵
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))

        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# 定义带位置编码的嵌入层
class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=5000):
        super(EmbeddingWithPositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return x


# 模拟无序的输入数据生成器
def generate_shuffled_input(vocab_size, seq_length, batch_size):
    input_data = torch.randint(0, vocab_size, (batch_size, seq_length))
    shuffled_data = input_data.clone()
    for i in range(batch_size):
        idx = list(range(seq_length))
        random.shuffle(idx)
        shuffled_data[i] = input_data[i, idx]
    return input_data, shuffled_data


# 参数设置
vocab_size = 100  # 词汇量大小
embed_size = 64  # 嵌入维度
seq_length = 10  # 序列长度
batch_size = 2  # 批大小

# 初始化带位置编码的嵌入层
embedding_layer = EmbeddingWithPositionalEncoding(vocab_size, embed_size, max_len=seq_length)

# 生成原始和无序输入数据
input_data, shuffled_data = generate_shuffled_input(vocab_size, seq_length, batch_size)

# 前向传播原始输入数据
original_output = embedding_layer(input_data)
print("原始输入的带位置编码嵌入输出:\n", original_output)

# 前向传播无序输入数据
shuffled_output = embedding_layer(shuffled_data)
print("无序输入的带位置编码嵌入输出:\n", shuffled_output)

# 比较原始与无序输入的嵌入输出差异
difference = torch.mean(torch.abs(original_output - shuffled_output))
print("原始与无序输入输出的平均差异:", difference.item())


# 进一步展示位置编码在生成语义上如何保持一致性
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, max_len):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = EmbeddingWithPositionalEncoding(vocab_size, embed_size, max_len)
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        attn_output, _ = self.attention(x, x, x)
        output = self.fc(attn_output)
        return output


# 设置模型参数
num_heads = 4  # 多头数量

# 初始化模型
model = SimpleTransformerModel(vocab_size, embed_size, num_heads, max_len=seq_length)

# 前向传播原始和无序输入数据
original_output = model(input_data)
shuffled_output = model(shuffled_data)

print("模型原始输入的输出:\n", original_output)
print("模型无序输入的输出:\n", shuffled_output)

# 比较模型在无序和原始输入下的语义差异
semantic_difference = torch.mean(torch.abs(original_output - shuffled_output))
print("模型在原始与无序输入下输出的平均语义差异:", semantic_difference.item())


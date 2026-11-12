import torch
import torch.nn as nn
import math


# 位置编码器的实现
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))

        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 增加批次维度
        self.register_buffer('pe', pe)  # 将pe存入缓冲区，不参与梯度更新

    def forward(self, x):
        # 将位置编码与输入嵌入相加
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


# 超参数设置
vocab_size = 100  # 词汇量大小
embed_size = 64  # 嵌入维度
max_len = 50  # 最大序列长度
batch_size = 2  # 批次大小

# 模拟输入数据
input_data = torch.randint(0, vocab_size, (batch_size, max_len))  # 随机生成词汇索引

# 初始化带位置编码的嵌入层
embedding_layer = EmbeddingWithPositionalEncoding(vocab_size, embed_size, max_len)

# 前向传播计算位置编码
output = embedding_layer(input_data)
print("输出的形状:", output.shape)
print("带位置编码的嵌入输出:\n", output)

# 检查位置编码的具体值
print("位置编码矩阵的部分内容:\n", embedding_layer.positional_encoding.pe[0, :10, :10])

# 测试位置编码在不同序列长度下的稳定性
test_input_data = torch.randint(0, vocab_size, (batch_size, 30))  # 较短序列
test_output = embedding_layer(test_input_data)
print("不同长度序列的输出形状:", test_output.shape)
print("带位置编码的嵌入输出（短序列）:\n", test_output)


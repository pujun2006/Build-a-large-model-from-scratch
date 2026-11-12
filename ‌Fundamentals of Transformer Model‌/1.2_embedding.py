import torch
import torch.nn as nn
import re
from collections import Counter
import random

# 设置随机种子
random.seed(42)
torch.manual_seed(42)


# 构建分词器，将文本转换为词汇索引
class Tokenizer:
    def __init__(self, texts, min_freq=1, max_vocab_size=10000):
        self.texts = texts
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.vocab = self.build_vocab()

    # 清洗并分割文本
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    # 构建词汇表
    def build_vocab(self):
        word_counts = Counter()
        for text in self.texts:
            tokens = self.preprocess(text)
            word_counts.update(tokens)
        # 保留频率高于 min_freq 的单词
        vocab = {"<pad>": 0, "<unk>": 1}
        for word, freq in word_counts.most_common(self.max_vocab_size - 2):
            if freq >= self.min_freq:
                vocab[word] = len(vocab)
        print(vocab)
        return vocab

    # 将文本转化为索引序列
    def text_to_sequence(self, text):
        tokens = self.preprocess(text)
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]


# 定义文本数据
texts = [
    "The quick brown fox jumps over the lazy dog",
    "An example of text tokenization for NLP",
    "PyTorch is widely used for deep learning tasks",
    "Natural language processing enables complex interactions",
]

# 实例化分词器
tokenizer = Tokenizer(texts)

# 转换示例文本为索引序列
text_sequence = tokenizer.text_to_sequence("The quick brown fox")
print("分词器输出的索引序列:", text_sequence)


# 定义嵌入层，将词汇索引转化为嵌入向量
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, x):
        return self.embedding(x)


# 超参数设置
VOCAB_SIZE = len(tokenizer.vocab)
EMB_DIM = 8  # 嵌入维度
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建嵌入层实例
embedding_layer = TextEmbedding(VOCAB_SIZE, EMB_DIM).to(DEVICE)

# 转换文本序列为张量并传递到嵌入层
text_tensor = torch.tensor([text_sequence], dtype=torch.long).to(DEVICE)
embedded_output = embedding_layer(text_tensor)

print("嵌入层输出的形状:", embedded_output.shape)
print("嵌入向量:\n", embedded_output)

# 训练循环示例（模拟训练嵌入）
optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 假设目标是另一个随机嵌入向量，模拟损失计算
target_embedding = torch.rand(embedded_output.shape).to(DEVICE)
loss = criterion(embedded_output, target_embedding)

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("训练后的损失:", loss.item())

# 显示嵌入层权重的部分
print("嵌入层的权重矩阵:\n", embedding_layer.embedding.weight[:5])

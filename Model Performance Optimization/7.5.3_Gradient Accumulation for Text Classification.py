from collections import Counter
from datasets import load_dataset
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer  # 如果后面不用可以删掉

# 超参数定义
batch_size = 32
accumulation_steps = 4  # 每4步累积梯度再更新
max_length = 200  # 限制文本最大长度
embedding_dim = 128
hidden_dim = 256
output_dim = 2
num_epochs = 2
lr = 0.001

# 1. 使用 Hugging Face 加载 IMDB 数据集
print("正在加载 IMDB 数据集...")
dataset = load_dataset("stanfordnlp/imdb")

# 2. 使用纯 Python 正则表达式替代 torchtext 的 get_tokenizer
# 效果：将文本转为小写，并切分成单词列表
tokenizer = lambda text: re.findall(r"\w+", text.lower())


# 构建一个简单的词表映射函数
def build_vocab(dataset_split, max_size=25000):
  counter = Counter()
  for item in dataset_split:
    counter.update(tokenizer(item["text"]))
  # 手动构建词表 (包含 <pad>=0, <unk>=1)
  vocab = {"<pad>": 0, "<unk>": 1}
  for word, _ in counter.most_common(max_size - 2):
    vocab[word] = len(vocab)
  return vocab


print("正在构建词表...")
vocab = build_vocab(dataset["train"])
vocab_size = len(vocab)
print(f"词表大小: {vocab_size}")


# 3. 自定义 PyTorch Dataset 适配 Hugging Face 数据集
class IMDBDataset(Dataset):

  def __init__(self, hf_dataset, vocab, tokenizer, max_length):
    self.data = hf_dataset
    self.vocab = vocab
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    text = item["text"]
    label = item["label"]  # 已经是 0 (neg) 或 1 (pos)

    # 分词与转数字 ID (未收录的词转为 <unk>=1)
    tokens = [self.vocab.get(token, 1) for token in self.tokenizer(text)]
    # 截断
    tokens = tokens[: self.max_length]

    return torch.tensor(tokens, dtype=torch.long), torch.tensor(
        label, dtype=torch.long
    )


train_dataset = IMDBDataset(dataset["train"], vocab, tokenizer, max_length)


# 4. 自定义 collate_fn 用于动态 Padding
def collate_batch(batch):
  labels = [item[1] for item in batch]
  texts = [item[0] for item in batch]
  # 使用 0 作为 padding 的默认值 (<pad> 对应的 id 是 0)
  texts_padded = nn.utils.rnn.pad_sequence(
      texts, batch_first=True, padding_value=0
  )
  return texts_padded, torch.stack(labels)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
)


# 5. 定义 LSTM 模型
class LSTMModel(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(LSTMModel, self).__init__()
    # padding_idx=0 告诉嵌入层遇到 0 时不计算梯度、输出全 0 向量
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    embedded = self.embedding(x)
    _, (hidden, _) = self.lstm(embedded)
    return self.fc(hidden[-1])


# 初始化模型、损失函数和优化器
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

total_batches = len(train_loader)

# 6. 训练循环（包含梯度累积）
print("开始训练...")
for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  optimizer.zero_grad()  # 每个 Epoch 开始前清空一次梯度

  for i, (texts, labels) in enumerate(train_loader):
    texts, labels = texts.cuda(), labels.cuda()

    # 前向传播
    outputs = model(texts)
    loss = criterion(outputs, labels)

    # 梯度累积（Loss 缩放）
    loss = loss / accumulation_steps
    loss.backward()

    running_loss += loss.item() * accumulation_steps

    # 判断是否触发更新（达到累计步数 或 已经是当前 Epoch 的最后一个 Batch）
    is_accumulation_step = (i + 1) % accumulation_steps == 0
    is_last_batch = (i + 1) == total_batches

    if is_accumulation_step or is_last_batch:
      optimizer.step()
      optimizer.zero_grad()

  print(f"Epoch {epoch + 1}, Loss: {running_loss / total_batches:.4f}")

# 7. 示例推理
model.eval()
with torch.no_grad():
  sample_text = "The movie was fantastic with great characters and plot"
  tokens = [vocab.get(token, 1) for token in tokenizer(sample_text)]
  tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).cuda()

  output = model(tokens_tensor)
  prediction = torch.argmax(output, dim=1).item()
  print(f"Sample Prediction (1: positive, 0: negative): {prediction}")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer


# 1. 定义 Prompt Tuning 模块
class PromptTuning(nn.Module):

  def __init__(self, prompt_length, embed_dim):
    super(PromptTuning, self).__init__()
    # 使用较小的标准差进行正态分布初始化
    self.prompt_embeddings = nn.Parameter(
        torch.randn(prompt_length, embed_dim) * 0.02
    )

  def forward(self, embedded_input):
    batch_size = embedded_input.size(0)
    prompt_embed = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat((prompt_embed, embedded_input), dim=1)


# 2. 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)


# 3. 数据准备函数
def prepare_data(texts, labels, tokenizer, max_length=128):
  encodings = tokenizer(
      texts,
      truncation=True,
      padding=True,
      max_length=max_length,
      return_tensors="pt",
  )
  input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
  labels = torch.tensor(labels)
  return TensorDataset(input_ids, attention_mask, labels)


# 4. 扩充后的测试数据集（增加样本多样性，避免极小样本过拟合）
texts = [
    "The model performs exceptionally well with prompt tuning.",
    "Prompt-based learning significantly enhances performance.",
    "This is a great, effective, and wonderful method.",
    "The performance is quite poor, bad, and disappointing.",
    "This approach completely fails to work properly.",
    "Terrible results achieved using this tuning strategy.",
]
labels = [1, 1, 1, 0, 0, 0]  #3正 3负

dataset = prepare_data(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Prompt Tuning 参数
prompt_length = 5
embed_dim = model.config.hidden_size


# 6. 集成到 BERT 模型中
class BertWithPromptTuning(nn.Module):

  def __init__(self, prompt_length, model):
    super(BertWithPromptTuning, self).__init__()
    self.bert = model
    self.prompt_length = prompt_length
    self.prompt_tuning = PromptTuning(prompt_length, embed_dim)

    # 核心：冻结 BERT 主干网络的所有参数，节省显存并仅训练 Prompt
    for param in self.bert.parameters():
      param.requires_grad = False

  def forward(self, input_ids, attention_mask):
    batch_size = input_ids.size(0)
    device = input_ids.device

    # 获取文本的原始 Embedding
    embedded_input = self.bert.bert.embeddings(input_ids)

    # 拼接 Prompt 嵌入
    embedded_input_with_prompt = self.prompt_tuning(embedded_input)

    # 扩展注意力掩码（在左侧拼接全 1 掩码）
    extended_attention_mask = torch.cat(
        [
            torch.ones((batch_size, self.prompt_length), device=device),
            attention_mask,
        ],
        dim=1,
    )

    # 显式构造并调整位置编码（Position IDs）
    seq_length = embedded_input_with_prompt.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    # 传入模型
    outputs = self.bert(
        inputs_embeds=embedded_input_with_prompt,
        attention_mask=extended_attention_mask,
        position_ids=position_ids,
    )

    return outputs.logits


# 7. 实例化模型与训练组件（调小学习率为 2e-5）
prompt_tuned_model = BertWithPromptTuning(prompt_length, model)
optimizer = optim.Adam(
    [{"params": prompt_tuned_model.prompt_tuning.parameters()}], lr=2e-5
)
criterion = nn.CrossEntropyLoss()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt_tuned_model.to(device)


# 8. 训练函数
def train_one_epoch(dataloader, model, criterion, optimizer):
  model.train()
  total_loss, correct_predictions = 0, 0

  for batch in dataloader:
    input_ids, attention_mask, labels = [x.to(device) for x in batch]

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    _, preds = torch.max(logits, dim=1)
    correct_predictions += torch.sum(preds == labels)

  avg_loss = total_loss / len(dataloader)
  accuracy = correct_predictions.double() / len(dataloader.dataset)
  return avg_loss, accuracy


# 9. 运行训练（增加到 5 个 Epoch 观察平稳收敛趋势）
for epoch in range(5):
  avg_loss, accuracy = train_one_epoch(
      dataloader, prompt_tuned_model, criterion, optimizer
  )
  print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

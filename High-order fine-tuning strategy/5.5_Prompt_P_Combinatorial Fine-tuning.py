import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer


# ==========================================
# 1. P-Tuning 编码器
# ==========================================
class PTuningEncoder(nn.Module):

  def __init__(self, prompt_length, embed_dim):
    super().__init__()
    self.prompt_length = prompt_length
    self.embedding = nn.Embedding(prompt_length, embed_dim)
    self.lstm = nn.LSTM(
        input_size=embed_dim,
        hidden_size=embed_dim,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
    )
    self.mlp = nn.Sequential(
        nn.Linear(embed_dim * 2, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
    )

  def forward(self, batch_size):
    device = self.embedding.weight.device
    input_ids = torch.arange(self.prompt_length, device=device).unsqueeze(0)
    embeds = self.embedding(input_ids).expand(batch_size, -1, -1)
    lstm_out, _ = self.lstm(embeds)
    prompt_embeds = self.mlp(lstm_out)
    return prompt_embeds


# ==========================================
# 2. 集成模型定义
# ==========================================
class BertWithPTuning(nn.Module):

  def __init__(self, prompt_length, model):
    super().__init__()
    self.bert = model
    embed_dim = self.bert.config.hidden_size
    self.ptuning_encoder = PTuningEncoder(prompt_length, embed_dim)

  def forward(self, input_ids, attention_mask, labels=None):
    batch_size = input_ids.size(0)
    inputs_embeds = self.bert.get_input_embeddings()(input_ids)
    prompt_embeds = self.ptuning_encoder(batch_size)

    # 拼接 Prompt 与文本嵌入
    combined_embeds = torch.cat((prompt_embeds, inputs_embeds), dim=1)

    # 扩展注意力掩码
    prompt_length = prompt_embeds.size(1)
    extended_attention_mask = torch.cat(
        [
            torch.ones((batch_size, prompt_length), device=attention_mask.device),
            attention_mask,
        ],
        dim=1,
    )

    outputs = self.bert(
        inputs_embeds=combined_embeds,
        attention_mask=extended_attention_mask,
        labels=labels,
    )
    return outputs


# ==========================================
# 3. 主程序：加载、训练与测试
# ==========================================
if __name__ == "__main__":
  # 【修改点 1】使用中文模型和分词器
  model_name = "bert-base-chinese"
  print(f"正在加载 {model_name}...")
  tokenizer = BertTokenizer.from_pretrained(model_name)
  base_model = BertForSequenceClassification.from_pretrained(
      model_name, num_labels=2
  )

  prompt_length = 5
  tuned_model = BertWithPTuning(prompt_length, base_model)

  # 准备训练文本和标签（1 代表 Positive）
  texts = [
      "BERT模型在这篇文章中有显著提升。",
      "P-Tuning在实际任务中表现很好。",
  ]
  labels = torch.tensor([1, 1])

  encodings = tokenizer(
      texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
  )
  input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]

  # 【修改点 2】P-Tuning 核心：冻结 BERT 主体参数，只训练 Prompt 编码器
  for param in tuned_model.bert.parameters():
    param.requires_grad = False

  # 只优化 P-Tuning 编码器的参数
  optimizer = AdamW(tuned_model.ptuning_encoder.parameters(), lr=1e-3)

  # 【修改点 3】增加一个简单的训练循环（让模型通过学习输出 Positive）
  print("\n开始进行几轮微调训练...")
  tuned_model.train()
  for epoch in range(15):
    optimizer.zero_grad()
    outputs = tuned_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/15 - Loss: {loss.item():.4f}")

  # 训练完成后进行推理
  print("\n--- 训练后的预测结果 ---")
  tuned_model.eval()
  with torch.no_grad():
    outputs = tuned_model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)

  for i, text in enumerate(texts):
    label_str = "Positive" if predictions[i].item() == 1 else "Negative"
    print(f"Text: {text}\nPrediction: {label_str}\n")
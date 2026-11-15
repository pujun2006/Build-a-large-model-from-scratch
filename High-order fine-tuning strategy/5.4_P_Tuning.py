import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. 定义真正的 P-Tuning 模块（包含 Prompt Encoder）
# ==========================================
class PTuningEncoder(nn.Module):
    def __init__(self, prompt_length, embed_dim, hidden_dim=768):
        super(PTuningEncoder, self).__init__()
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim

        # 定义底层可学习的原始嵌入（虚拟 Token 的基础）
        self.raw_embeddings = nn.Embedding(prompt_length, embed_dim)

        # P-Tuning 的核心：Prompt Encoder (使用 MLP 重参数化，平滑优化空间)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, batch_size, device):
        # 1. 生成虚拟 Token 的索引：[0, 1, 2, ..., prompt_length-1]
        prompt_ids = torch.arange(self.prompt_length, dtype=torch.long, device=device)
        prompt_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, prompt_length]

        # 2. 查表获取原始嵌入
        inputs_embeds = self.raw_embeddings(prompt_ids)  # [batch_size, prompt_length, embed_dim]

        # 3. 通过 MLP 生成最终的连续 Prompt 嵌入
        encoded_embeds = self.mlp(inputs_embeds)

        return encoded_embeds


# ==========================================
# 2. 将 P-Tuning 集成到 BERT 模型中
# ==========================================
class BertWithRealPTuning(nn.Module):
    def __init__(self, prompt_length, embed_dim, model):
        super(BertWithRealPTuning, self).__init__()
        self.bert = model
        self.prompt_length = prompt_length

        # 核心：冻结 BERT 主干网络的所有参数，只训练 P-Tuning 模块
        for param in self.bert.parameters():
            param.requires_grad = False

        # 实例化 P-Tuning 的 Prompt Encoder 模块
        self.p_tuning_encoder = PTuningEncoder(prompt_length, embed_dim)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # 1. 获取真实文本的词嵌入
        text_embeddings = self.bert.bert.embeddings(input_ids)  # [batch_size, seq_len, embed_dim]

        # 2. 通过 Prompt Encoder 生成连续的虚拟 Prompt 嵌入
        prompt_embeddings = self.p_tuning_encoder(batch_size, device)  # [batch_size, prompt_length, embed_dim]

        # 3. 将虚拟 Prompt 拼接到真实文本前面：[Prompt ; Text]
        combined_embeddings = torch.cat((prompt_embeddings, text_embeddings), dim=1)

        # 4. 扩展注意力掩码（Attention Mask），让模型关注到新增的 Prompt 部分
        extended_attention_mask = torch.cat([
            torch.ones((batch_size, self.prompt_length), device=device),
            attention_mask
        ], dim=1)

        # 5. 传入冻结的 BERT 主干网络
        outputs = self.bert(
            inputs_embeds=combined_embeddings,
            attention_mask=extended_attention_mask
        )

        return outputs.logits


# ==========================================
# 3. 主程序：初始化、数据准备与训练循环
# ==========================================
if __name__ == "__main__":
    # 自动检测设备 (GPU / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载分词器和基础模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # 准备稍微充实一点的示例数据（避免极小样本造成的剧烈震荡）
    texts = [
        "The model performs exceptionally well with P-Tuning.",
        "Prompt-based learning enhances performance.",
        "This is a fantastic approach for few-shot learning.",
        "The results are terrible and completely incorrect."
    ]
    labels = [1, 1, 1, 0]

    # 数据编码与打包
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors="pt")
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 超参数配置
    prompt_length = 5
    embed_dim = base_model.config.hidden_size

    # 实例化包装后的 P-Tuning 模型并移动到指定设备
    p_tuned_model = BertWithRealPTuning(prompt_length, embed_dim, base_model)
    p_tuned_model.to(device)

    # 定义优化器（只优化 p_tuning_encoder 中的参数）和损失函数
    optimizer = optim.AdamW(p_tuned_model.p_tuning_encoder.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 开始训练循环
    epochs = 5
    print("\n开始 P-Tuning 训练...")

    p_tuned_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]

            optimizer.zero_grad()

            # 前向传播
            logits = p_tuned_model(input_ids, attention_mask)

            # 计算损失
            loss = criterion(logits, batch_labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    print("\n训练结束！P-Tuning 模块运行正常。")

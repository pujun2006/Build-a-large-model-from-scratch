import torch
from torch import nn, optim
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# 假设已有预处理后的数据集
X_train = ["The movie was great!", "I did not like the film.", "An excellent experience."]
X_labels = [1, 0, 1]  # 假设标签为 1 表示正面，0 表示负面

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True, max_length=32)
labels = torch.tensor(X_labels)

train_data = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
train_loader = DataLoader(train_data, batch_size=2)

# 加载教师模型（预训练的 BERT）
teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
teacher_model.eval()  # 设为评估模式，不更新教师模型的权重

# 加载学生模型（DistilBERT）
student_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
student_model.train()  # 设为训练模式

# 1. 补充定义优化器
optimizer = optim.Adam(student_model.parameters(), lr=5e-5)


def distillation_loss(student_logits, teacher_logits, true_labels, alpha=0.5, temperature=2.0):
    # 教师模型输出使用 softmax 转为概率分布
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    # 学生模型输出使用 log_softmax 转为对数概率
    student_log_probs = nn.functional.log_softmax(student_logits / temperature, dim=-1)

    # 软标签的 KL 散度损失 (乘以 T^2 保持梯度平稳)
    kd_loss = nn.functional.kl_div(student_log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)

    # 使用真实标签的交叉熵损失
    ce_loss = nn.CrossEntropyLoss()(student_logits, true_labels)

    # 2. 修复了漏掉的减号
    return alpha * kd_loss + (1 - alpha) * ce_loss


epochs = 3
for epoch in range(epochs):
    epoch_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, batch_labels = batch

        # 教师模型预测
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        # 学生模型预测
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        # 计算蒸馏损失
        loss = distillation_loss(student_logits, teacher_logits, batch_labels)
        epoch_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
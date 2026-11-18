import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

# 1. 加载BERT的教师模型和分词器
teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 2. 构建真正的学生模型：通过修改 Config 来减少层数
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # 获取基础配置并修改层数
        student_config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
        student_config.num_hidden_layers = 4  # 缩减BERT层数为4层
        # 使用该配置初始化学生模型（通常为随机初始化权重）
        self.bert = BertForSequenceClassification(student_config)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)


student_model = StudentModel()


# 3. 修正后的蒸馏损失函数
def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    loss_fn = nn.CrossEntropyLoss()

    # 软标签损失（软化概率分布），使用 batchmean 保证数学严谨性
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / T, dim=1),
        nn.functional.softmax(teacher_logits / T, dim=1)
    ) * (T * T)

    # 硬标签损失（真实标签交叉熵）
    hard_loss = loss_fn(student_logits, labels)

    # 修复了漏掉的减号
    return alpha * soft_loss + (1 - alpha) * hard_loss


# 4. 假设一个简单的数据集
texts = ["This is a positive example.", "This is a negative example."]
labels = torch.tensor([1, 0])

# 数据预处理
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 优化器设置
optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

# 5. 蒸馏训练过程
epochs = 3
for epoch in range(epochs):
    student_model.train()
    teacher_model.eval()

    # 教师模型输出
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_outputs.logits

    # 学生模型输出
    student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = student_outputs.logits

    # 计算蒸馏损失
    loss = distillation_loss(student_logits, teacher_logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
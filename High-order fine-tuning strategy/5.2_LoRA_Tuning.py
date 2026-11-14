import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset


# 1. 低秩残差模块定义（类似 LoRA 的低秩矩阵思路）
class LoRA(nn.Module):
    def __init__(self, input_dim, rank=4):
        super(LoRA, self).__init__()
        self.rank = rank
        self.down_proj = nn.Linear(input_dim, rank, bias=False)  # 降维矩阵
        self.up_proj = nn.Linear(rank, input_dim, bias=False)  # 升维矩阵

    def forward(self, x):
        return self.up_proj(self.down_proj(x))


# 2. 带有低秩模块和分类头的 BERT 模型
class BertWithLoRA(nn.Module):
    def __init__(self, lora_rank=4, num_classes=2):
        super(BertWithLoRA, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 冻结 BERT 原有参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 为每层增加低秩适应模块
        self.lora_modules = nn.ModuleList([
            LoRA(self.bert.config.hidden_size, lora_rank)
            for _ in range(self.bert.config.num_hidden_layers)
        ])

        # 【新增】分类头：将 768 维映射为分类类别数（2类）
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # 【修复】将 tuple 转为 list，以便能够修改各层隐状态
        hidden_states = list(outputs.hidden_states)

        # 在每层应用低秩模块进行残差修正
        for i, lora in enumerate(self.lora_modules):
            hidden_states[i + 1] = hidden_states[i + 1] + lora(hidden_states[i + 1])

        # 取出最后一层 [CLS] 标记的输出 (形状: [batch_size, 768])
        cls_output = hidden_states[-1][:, 0, :]

        # 通过分类头得到 logits (形状: [batch_size, num_classes])
        logits = self.classifier(cls_output)
        return logits


# 3. 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 4. 准备示例数据
def prepare_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_mask, labels)


texts = ["LoRA efficiently adapts large models.", "This technique uses low-rank updates."]
labels = [1, 0]
dataset = prepare_data(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 5. 实例化模型与训练组件
model = BertWithLoRA(lora_rank=4, num_classes=2)

# 【修复】优化器需要同时优化 lora 模块参数 和 分类头参数
optimizer = optim.Adam(
    list(model.lora_modules.parameters()) + list(model.classifier.parameters()),
    lr=1e-4
)
criterion = nn.CrossEntropyLoss()

# 训练过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_one_epoch(dataloader, model, criterion, optimizer):
    model.train()
    total_loss, correct_predictions = 0, 0

    for batch in dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        # 前向传播直接返回 logits
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失与准确率
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_loss, accuracy


# 6. 运行训练
for epoch in range(3):
    avg_loss, accuracy = train_one_epoch(dataloader, model, criterion, optimizer)
    # 【修复】修正了 print 语句的换行和格式错误
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
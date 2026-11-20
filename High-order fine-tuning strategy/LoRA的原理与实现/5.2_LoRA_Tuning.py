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

        hidden_size = self.bert.config.hidden_size
        num_layers = self.bert.config.num_hidden_layers

        # 为每层增加低秩适应模块
        self.lora_modules = nn.ModuleList([
            LoRA(hidden_size, lora_rank)
            for _ in range(num_layers)
        ])

        # 分类头：将 768 维映射为分类类别数（2类）
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 开启 output_hidden_states=True 获取每一层的输出隐状态
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # 在特征流动过程中逐层累加 LoRA 修正量
        current_hidden = hidden_states[0]

        for i, lora in enumerate(self.lora_modules):
            layer_out = hidden_states[i + 1]
            current_hidden = layer_out + lora(layer_out)

        # 取出最后一层经过修正后的 [CLS] 标记输出 (形状: [batch_size, 768])
        cls_output = current_hidden[:, 0, :]

        # 通过分类头得到 logits (形状: [batch_size, num_classes])
        logits = self.classifier(cls_output)
        return logits


# 3. 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 4. 准备示例数据（适当扩充样本，使得准确率变化更平滑）
def prepare_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_mask, labels)


texts = [
    "LoRA efficiently adapts large models.",
    "This technique uses low-rank updates.",
    "The weather is very nice today.",
    "I love programming in Python."
]
labels = [1, 1, 0, 0]
dataset = prepare_data(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. 实例化模型与训练组件
model = BertWithLoRA(lora_rank=4, num_classes=2)

# 优化器同时优化 lora 模块参数 和 分类头参数
optimizer = optim.Adam(
    list(model.lora_modules.parameters()) + list(model.classifier.parameters()),
    lr=1e-4
)
criterion = nn.CrossEntropyLoss()

# 训练过程设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_one_epoch(dataloader, model, criterion, optimizer):
    model.train()
    total_loss, correct_predictions = 0, 0

    for batch in dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        # 前向传播
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        # 反向传播与优化
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
for epoch in range(5):
    avg_loss, accuracy = train_one_epoch(dataloader, model, criterion, optimizer)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

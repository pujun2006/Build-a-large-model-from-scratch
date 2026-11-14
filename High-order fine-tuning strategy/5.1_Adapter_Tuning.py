import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset


# 1. Adapter模块的定义
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        x_down = self.down_project(x)
        x_activated = self.activation(x_down)
        x_up = self.up_project(x_activated)
        return x + x_up  # 残差连接


# 2. 带有 Adapter 和分类头的 BERT 模型
class BertWithAdapter(nn.Module):
    def __init__(self, adapter_dim=64, num_classes=2):
        super(BertWithAdapter, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 冻结 BERT 原有参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 为每一层 Transformer 创建一个 Adapter 模块
        self.adapters = nn.ModuleList(
            [Adapter(self.bert.config.hidden_size, adapter_dim) for _ in range(self.bert.config.num_hidden_layers)]
        )

        # 【新增】分类头：将 BERT 最后的 768 维 [CLS] 特征映射为类别数（这里是 2 类）
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states)

        # 逐层注入 Adapter
        for i in range(len(self.adapters)):
            delta = self.adapters[i](hidden_states[i + 1])
            hidden_states[i + 1] = hidden_states[i + 1] + delta

        # 取出最后一层 [CLS] 标记的向量 (形状: [batch_size, 768])
        cls_output = hidden_states[-1][:, 0, :]

        # 通过分类头映射为 logits (形状: [batch_size, num_classes])
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


texts = ["The model is effective for tuning.", "Adapters enable efficient fine-tuning."]
labels = [1, 0]
dataset = prepare_data(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 5. 实例化模型与训练组件
model = BertWithAdapter(adapter_dim=64, num_classes=2)

# 【关键修改】优化器需要同时优化 adapter 参数 和 分类头 (classifier) 参数
optimizer = optim.Adam(
    list(model.adapters.parameters()) + list(model.classifier.parameters()),
    lr=1e-4
)
criterion = nn.CrossEntropyLoss()

# 训练设备配置
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
for epoch in range(3):
    avg_loss, accuracy = train_one_epoch(dataloader, model, criterion, optimizer)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
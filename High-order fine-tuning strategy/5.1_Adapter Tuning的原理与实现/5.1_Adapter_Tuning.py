import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset


# 1. Adapter 模块的定义
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


# 2. 层的包装器：兼容返回元组或纯 Tensor 的不同 Transformers 版本
class AdapterWrappedBertLayer(nn.Module):
    def __init__(self, original_layer, adapter_dim=64):
        super(AdapterWrappedBertLayer, self).__init__()
        self.layer = original_layer
        # 获取 hidden_size
        hidden_size = self.layer.output.dense.out_features
        self.adapter = Adapter(hidden_size, adapter_dim)

    def forward(self, hidden_states, *args, **kwargs):
        # 调用原 BERT 层
        layer_outputs = self.layer(hidden_states, *args, **kwargs)

        # 兼容处理：如果返回的是 tuple，则提取第一个元素并保留其余返回值
        if isinstance(layer_outputs, tuple):
            output_tensor = layer_outputs[0]
            adapted_tensor = self.adapter(output_tensor)
            return (adapted_tensor,) + layer_outputs[1:]
        else:
            # 如果返回的是纯 Tensor
            adapted_tensor = self.adapter(layer_outputs)
            return adapted_tensor


# 3. 带有 Adapter 和分类头的 BERT 模型
class BertWithAdapter(nn.Module):
    def __init__(self, adapter_dim=64, num_classes=2):
        super(BertWithAdapter, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 1. 冻结 BERT 原有参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 2. 用包装了 Adapter 的层原地替换 BERT 原有的 encoder layer
        hidden_size = self.bert.config.hidden_size
        for i, layer in enumerate(self.bert.encoder.layer):
            self.bert.encoder.layer[i] = AdapterWrappedBertLayer(layer, adapter_dim)

        # 3. 分类头
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # 直接调用 BERT 模型，内部会顺畅执行我们替换后的包装层
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 获取最终输出的 [CLS] 向量 (形状: [batch_size, 768])
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 通过分类头映射为 logits
        logits = self.classifier(cls_output)
        return logits


# 4. 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 5. 准备示例数据
def prepare_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_mask, labels)


texts = ["The model is effective for tuning.", "Adapters enable efficient fine-tuning."]
labels = [1, 0]
dataset = prepare_data(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# 6. 实例化模型与训练组件
model = BertWithAdapter(adapter_dim=64, num_classes=2)

# 收集所有需要训练的参数（遍历模型中所有的 Adapter 以及分类头）
adapter_params = [p for n, p in model.named_parameters() if "adapter" in n or "classifier" in n]
optimizer = optim.Adam(adapter_params, lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


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
        correct_predictions += torch.sum(preds == labels).item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_loss, accuracy


# 7. 运行训练
for epoch in range(3):
    avg_loss, accuracy = train_one_epoch(dataloader, model, criterion, optimizer)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

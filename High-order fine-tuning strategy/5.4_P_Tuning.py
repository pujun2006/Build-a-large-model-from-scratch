import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset


# 定义P-Tuning模块，用于生成可训练的Prompt嵌入
class PTuning(nn.Module):
    def __init__(self, prompt_length, embed_dim):
        super(PTuning, self).__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_length, embed_dim))

    def forward(self, embedded_input):
        batch_size = embedded_input.size(0)
        prompt_embed = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat((prompt_embed, embedded_input), dim=1)


# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# 数据准备函数
def prepare_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_mask, labels)


# 定义示例数据
texts = ["The model performs well with prompt tuning.", "Prompt-based learning enhances performance."]
labels = [1, 0]
dataset = prepare_data(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

# P-Tuning参数
prompt_length = 5  # 自定义Prompt长度
embed_dim = model.config.hidden_size


# 将P-Tuning集成到BERT模型中
class BertWithPTuning(nn.Module):
    def __init__(self, prompt_length, model):
        super(BertWithPTuning, self).__init__()
        self.bert = model

        # 【关键修正 1】冻结 BERT 主干网络的所有参数，节省显存并防止被误训练
        for param in self.bert.parameters():
            param.requires_grad = False

        self.p_tuning = PTuning(prompt_length, embed_dim)

    def forward(self, input_ids, attention_mask):
        embedded_input = self.bert.bert.embeddings(input_ids)
        embedded_input_with_prompt = self.p_tuning(embedded_input)

        # 扩展注意力掩码以匹配Prompt长度
        extended_attention_mask = torch.cat(
            [torch.ones((attention_mask.size(0), prompt_length), device=attention_mask.device), attention_mask], dim=1)
        outputs = self.bert(inputs_embeds=embedded_input_with_prompt, attention_mask=extended_attention_mask)

        return outputs.logits


# 实例化模型与训练组件
p_tuned_model = BertWithPTuning(prompt_length, model)
optimizer = optim.Adam([{'params': p_tuned_model.p_tuning.parameters()}], lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 训练过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p_tuned_model.to(device)


def train_one_epoch(dataloader, model, criterion, optimizer):
    model.train()
    total_loss, correct_predictions = 0, 0

    for batch in dataloader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        # 前向传播
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    return avg_loss, accuracy


# 运行训练
for epoch in range(3):
    avg_loss, accuracy = train_one_epoch(dataloader, p_tuned_model, criterion, optimizer)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
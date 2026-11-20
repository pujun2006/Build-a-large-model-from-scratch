import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset

# 检查并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用设备: {device}")


# 创建用户评价数据集（简单示例）
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=128,
                                return_tensors='pt')
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), torch.tensor(self.labels[idx])


# 简单用户评价样本与标签
texts = ["This product is great!", "Worst purchase ever.", "Not bad, but could be better.", "I absolutely love it!",
         "Terrible experience."]
labels = [1, 0, 1, 1, 0]
dataset = TextDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# 定义BERT分类器（支持接收 inputs_embeds）
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 二分类

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # 兼容标准的 input_ids 输入或对抗训练的 inputs_embeds 输入
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token的输出
        return self.classifier(cls_output)


# 初始化模型、损失函数与优化器
model = BertClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)


# FGM 对抗训练辅助类
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # 寻找嵌入层梯度并施加对抗扰动
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # 恢复模型的原始嵌入参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


fgm = FGM(model)

# 对抗训练过程
print("开始训练...")
for epoch in range(5):
    for input_ids, attention_mask, labels in dataloader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        # 1. 正常前向传播与损失计算
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        # 2. 对正常损失进行反向传播，计算出嵌入层的梯度
        loss.backward(retain_graph=True)

        # 3. 对抗训练：在嵌入空间添加扰动
        fgm.attack(epsilon=1.0, emb_name='word_embeddings')

        # 4. 再次前向传播（基于对抗扰动后的嵌入）
        embedding_output = model.bert.embeddings(input_ids)
        adv_outputs = model(inputs_embeds=embedding_output, attention_mask=attention_mask)
        adv_loss = criterion(adv_outputs, labels)

        # 5. 对抗损失反向传播
        adv_loss.backward()

        # 6. 恢复原始嵌入参数，并更新优化器
        fgm.restore(emb_name='word_embeddings')
        optimizer.step()

    print(f'Epoch {epoch + 1}/5, Loss: {(loss + adv_loss).item() / 2:.4f}')


# 预测函数
def predict(text):
    model.eval()
    tokens = dataset.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input_ids, attention_mask = tokens['input_ids'].to(device), tokens['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(output, dim=1)
    return "Positive" if prediction.item() == 1 else "Negative"


# 测试模型表现
test_text = "I enjoyed the product, but it has some issues."
print("-" * 30)
print("测试文本:", test_text)
print("模型预测结果:", predict(test_text))

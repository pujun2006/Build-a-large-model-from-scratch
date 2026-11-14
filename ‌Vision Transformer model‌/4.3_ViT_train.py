# 以下代码建议在GPU上运行
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import numpy as np


# 定义图像分块和嵌入模块
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        # 序列长度 = (img_size / patch_size)^2 + 1 (cls_token)
        num_patches = (img_size // patch_size) ** 2
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))

    def forward(self, x):
        # x shape: (B, C, H, W) -> (B, embed_size, H/P, W/P) -> (B, embed_size, num_patches) -> (B, num_patches, embed_size)
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        return x


# Transformer编码器块
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        # 显式指定 batch_first=True，适配 (B, Seq_Len, Embed_Dim) 的输入形状
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),  # ViT 通常使用 GELU 激活函数，效果比 ReLU 更好
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN Transformer 结构（现代 Transformer 常规做法）
        norm_x = self.norm1(x)
        attention = self.attention(norm_x, norm_x, norm_x)[0]
        x = x + self.dropout(attention)

        forward = self.feed_forward(self.norm2(x))
        out = x + forward
        return out


# Vision Transformer模型定义
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_size=768, num_heads=8, num_layers=6,
                 forward_expansion=4, dropout=0.1, num_classes=10):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size, img_size)
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.transformer:
            x = layer(x)
        # 取出分类 Token (cls_token) 的输出用于分类
        cls_token_out = x[:, 0]
        return self.mlp_head(cls_token_out)


# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 增加图像归一化
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 模型、损失函数和优化器
model = VisionTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


# 训练函数
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)


# 验证函数
def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    return total_loss / len(dataloader), correct / len(dataloader.dataset)


# 训练和验证过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epochs = 5
for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
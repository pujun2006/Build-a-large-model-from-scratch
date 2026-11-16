import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义两种不同的数据增强，用于生成正样本对（视图1和视图2）
transform1 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform2 = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor()
])


# 自定义对比学习数据集：返回 锚点(Anchor)、正样本(Positive)、负样本(Negative)
class ContrastiveDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # 注意：这里的 dataset 初始化时不要带 transform，返回原始 PIL Image

    def __getitem__(self, index):
        img, _ = self.dataset[index]

        # 同一张图做两次不同增强 -> 正样本对（Anchor & Positive）
        anchor = transform1(img)
        positive = transform2(img)

        # 随机挑选另一张不同的图 -> 负样本 (Negative)
        neg_index = np.random.choice(len(self.dataset))
        while neg_index == index:
            neg_index = np.random.choice(len(self.dataset))
        neg_img, _ = self.dataset[neg_index]
        negative = transform1(neg_img)

        return anchor, positive, negative

    def __len__(self):
        return len(self.dataset)


# 构建数据集（不带 transform，确保拿到的是 PIL Image）
raw_dataset = CIFAR10(root='./data', train=True, download=True)
contrastive_dataset = ContrastiveDataset(raw_dataset)
dataloader = DataLoader(contrastive_dataset, batch_size=32, shuffle=True)


# 模型定义（修正了全连接层输入维度为 128 * 8 * 8 = 8192）
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256)
        )

    def forward(self, x):
        return self.conv(x)


# 正确的对比损失（三元组损失 Triplet Loss 范式）
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算欧氏距离
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative, p=2)

        # 目标：让 pos_dist 尽可能小，neg_dist 尽可能大
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


# 初始化模型、损失函数和优化器
encoder = Encoder()
contrastive_loss = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(encoder.parameters(), lr=0.001)

# 训练过程
for epoch in range(5):
    total_loss = 0
    for anchor_img, positive_img, negative_img in dataloader:
        anchor_emb = encoder(anchor_img)
        positive_emb = encoder(positive_img)
        negative_emb = encoder(negative_img)

        # 计算损失并反向传播
        loss = contrastive_loss(anchor_emb, positive_emb, negative_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

print("对比学习训练完成！")
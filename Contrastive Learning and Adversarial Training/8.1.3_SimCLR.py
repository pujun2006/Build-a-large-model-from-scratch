import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 数据增强：包括随机裁剪、颜色抖动、水平翻转等
transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.ToTensor()
])


# SimCLR 数据集
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

    def __len__(self):
        return len(self.dataset)


# CIFAR-10数据集
dataset = CIFAR10(root='./data', train=True, download=True)
simclr_dataset = SimCLRDataset(dataset)
dataloader = DataLoader(simclr_dataset, batch_size=64, shuffle=True, drop_last=True)


# 简单的编码器网络（已修正语法错误）
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


# 投影头
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 标准高效的 NT-Xent 对比损失（基于 2N 矩阵）
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)

        # 1. L2 归一化投影特征
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 2. 拼接两个视图的特征：形状变为 (2N, Dim)
        representations = torch.cat([z_i, z_j], dim=0)

        # 3. 计算全量相似度矩阵 (2N, 2N)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        # 4. 构造对称的目标标签
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
        labels = labels.to(z_i.device)

        # 5. 屏蔽自身与自身的相似度（对角线设为负无穷）
        mask = torch.eye(2 * batch_size, dtype=bool).to(z_i.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))

        # 6. 计算交叉熵损失
        loss = self.criterion(similarity_matrix, labels)
        return loss


# 模型、损失和优化器定义
encoder = Encoder()
projection_head = ProjectionHead()
criterion = NTXentLoss(temperature=0.5)
optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=0.001)

# 训练循环
for epoch in range(5):  # 示例设置为5个epoch
    epoch_loss = 0
    for img1, img2 in dataloader:
        z_i = projection_head(encoder(img1))
        z_j = projection_head(encoder(img2))

        # 计算对比损失
        loss = criterion(z_i, z_j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/5], Loss: {epoch_loss / len(dataloader):.4f}")

print("SimCLR训练完成，模型可以正常收敛。")
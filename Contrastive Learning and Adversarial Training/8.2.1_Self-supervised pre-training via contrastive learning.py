import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.ToTensor()
])


# 定义自监督数据集
class ContrastiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

    def __len__(self):
        return len(self.dataset)


# 加载数据集（建议加上 drop_last=True 保证每个 Batch 大小一致）
dataset = datasets.CIFAR10(root="./data", download=True)
contrastive_dataset = ContrastiveLearningDataset(dataset)
dataloader = DataLoader(contrastive_dataset, batch_size=64, shuffle=True, drop_last=True)


# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # 输入尺寸 32x32 -> 经过三次 stride=2 的卷积后变为 4x4
        # 通道数 256，因此展平后大小为 256 * 4 * 4 = 4096
        self.fc = nn.Linear(256 * 4 * 4, 128)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 定义投影头网络
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=64):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


# 定义标准的 NT-Xent 对比损失（采用标准 2N 矩阵与对称交叉熵）
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

        # 4. 构造对称的目标标签 (大小为 2N)
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0)
        labels = labels.to(z_i.device)

        # 5. 屏蔽自身与自身的相似度（将对角线设为一个极小值，避免模型自我匹配）
        mask = torch.eye(2 * batch_size, dtype=bool).to(z_i.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))

        # 6. 计算交叉熵损失
        loss = self.criterion(similarity_matrix, labels)
        return loss


# 初始化模型和损失
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
projection_head = ProjectionHead().to(device)
criterion = NTXentLoss(temperature=0.5)
optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=1e-3)

# 开始训练
encoder.train()
projection_head.train()
for epoch in range(5):  # 示例为5个epoch
    epoch_loss = 0
    for img1, img2 in dataloader:
        img1, img2 = img1.to(device), img2.to(device)

        # 获取编码器特征
        h_i = encoder(img1)
        h_j = encoder(img2)

        # 生成投影向量
        z_i = projection_head(h_i)
        z_j = projection_head(h_j)

        # 计算对比损失
        loss = criterion(z_i, z_j)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/5], Loss: {epoch_loss / len(dataloader):.4f}")

print("对比学习自监督预训练完成。")
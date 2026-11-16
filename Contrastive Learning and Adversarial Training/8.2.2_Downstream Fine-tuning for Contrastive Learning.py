import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# 假设前面的 Encoder 类已经定义过
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
        self.fc = nn.Linear(256 * 4 * 4, 128)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 定义数据增强（微调/测试时通常只做 Resize 和 ToTensor，也可以加上标准化）
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 加载CIFAR-10数据集
dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_size = int(0.8 * len(dataset))  # 修复点：补全了乘号 *
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# 定义分类器网络，使用预训练的编码器
class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, num_classes)  # 添加线性分类头

    def forward(self, x):
        with torch.no_grad():  # 冻结编码器特征提取过程
            features = self.encoder(x)
        return self.fc(features)


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化预训练编码器（实际应用中，这里应该通过 torch.load 加载预训练权重）
pretrained_encoder = Encoder()
# 示例：pretrained_encoder.load_state_dict(torch.load("simclr_encoder.pth"))
pretrained_encoder = pretrained_encoder.to(device)

# 实例化分类器并移动到设备
classifier = Classifier(pretrained_encoder, num_classes=10).to(device)

# 只对分类头（fc）的参数进行优化（线性探测的标准做法）
optimizer = optim.Adam(classifier.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# 定义验证过程
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total  # 修复点：补全乘号 *
    return val_loss / len(val_loader), accuracy


# 定义微调过程
def fine_tune(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device=device):
    for epoch in range(num_epochs):
        model.train()  # 训练模式
        total_loss = 0
        correct = 0
        total = 0

        # 训练循环
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 修复点：数据送入设备

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100.0 * correct / total  # 修复点：补全乘号 *

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Loss: {total_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")


# 开始线性探测评估
fine_tune(classifier, train_loader, val_loader, criterion, optimizer, num_epochs=5)
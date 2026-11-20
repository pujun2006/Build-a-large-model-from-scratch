import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设置随机种子
torch.manual_seed(42)


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # 如果需要对维度进行调整
        if self.downsample is not None:
            identity = self.downsample(x)

        # 卷积操作并激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 将输入添加到输出
        out += identity
        out = self.relu(out)

        return out


# 构建简单的网络模型，包含多个残差块
class ResNetLike(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetLike, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 使用残差块构建网络
        self.layer2 = ResidualBlock(64, 64)
        self.layer3 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        ))
        self.layer4 = ResidualBlock(128, 128)

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 模拟输入
input_data = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels (RGB), 32x32 image size

# 创建模型
model = ResNetLike(num_classes=10)

# 前向传播
output = model(input_data)
print("网络输出:", output)

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟训练步骤
target = torch.tensor([1])  # 假设的目标类别
optimizer.zero_grad()
loss = criterion(output, target)
loss.backward()
optimizer.step()

print("训练损失:", loss.item())

# 检查残差块中的参数更新情况
for name, param in model.layer2.named_parameters():
    print(f"{name}: {param.grad}")

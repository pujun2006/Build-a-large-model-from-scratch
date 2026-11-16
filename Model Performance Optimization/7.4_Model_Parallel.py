# 因缺乏多GPU环境，以下模型并行处理的代码尚未进行验证
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, models, transforms
# from torchvision.models import ResNet18_Weights
#
# # 1. 数据预处理
# transform = transforms.Compose(
#     [transforms.Resize((224, 224)), transforms.ToTensor()]
# )
#
# # 使用 CIFAR-10 数据集
# train_dataset = datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform
# )
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
#
# # 2. 修正后的模型并行结构
# class ModelParallelResNet(nn.Module):
#
#   def __init__(self):
#     super(ModelParallelResNet, self).__init__()
#     # 使用现代 weights 参数加载预训练模型
#     resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
#
#     # 【前半部分】放在 cuda:0：包含初始卷积、池化以及 layer1、layer2
#     self.stage1 = nn.Sequential(
#         resnet.conv1,
#         resnet.bn1,
#         resnet.relu,
#         resnet.maxpool,
#         resnet.layer1,
#         resnet.layer2,
#     ).to("cuda:0")
#
#     # 【后半部分】放在 cuda:1：包含 layer3、layer4 以及 avgpool
#     self.stage2 = nn.Sequential(
#         resnet.layer3,
#         resnet.layer4,
#         resnet.avgpool,
#     ).to("cuda:1")
#
#     # 修改全连接层用于 10 分类，并放入 cuda:1
#     self.fc = nn.Linear(resnet.fc.in_features, 10).to("cuda:1")
#
#   def forward(self, x):
#     # 1. 确保输入在 cuda:0，并运行前半部分
#     x = x.to("cuda:0")
#     x = self.stage1(x)
#
#     # 2. 将中间激活值传送到 cuda:1，运行后半部分
#     x = x.to("cuda:1")
#     x = self.stage2(x)
#
#     # 3. 展平并通过全连接层
#     x = torch.flatten(x, 1)
#     x = self.fc(x)
#     return x
#
#
# # 初始化模型
# model = ModelParallelResNet()
# model.train()
#
# # 损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#
# # 3. 训练过程
# num_epochs = 2
# for epoch in range(num_epochs):
#   running_loss = 0.0
#   for i, (inputs, labels) in enumerate(train_loader):
#     # 输入送到 cuda:0，标签送到模型最终所在的 cuda:1
#     inputs, labels = inputs.to("cuda:0"), labels.to("cuda:1")
#
#     optimizer.zero_grad()
#
#     # 前向传播（输出在 cuda:1）
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#
#     # 反向传播与优化（PyTorch 自动处理跨 GPU 的梯度反向传播）
#     loss.backward()
#     optimizer.step()
#
#     running_loss += loss.item()
#
#   print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")




# 以下是单机兼容版
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# 1. 数据预处理
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

# 使用 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 自动检测可用设备（单卡或 CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")


# 2. 单卡兼容版网络结构（将所有层统一放在单张可用卡上）
class SingleDeviceResNet(nn.Module):

  def __init__(self):
    super(SingleDeviceResNet, self).__init__()
    resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # 将所有层组合并整体移至目标设备
    self.features = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet.avgpool,
    ).to(device)

    # 修改全连接层并移至目标设备
    self.fc = nn.Linear(resnet.fc.in_features, 10).to(device)

  def forward(self, x):
    x = x.to(device)
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x


# 初始化模型
model = SingleDeviceResNet()
model.train()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 3. 训练过程
num_epochs = 1  # 演示用，设为 1 个 epoch
for epoch in range(num_epochs):
  running_loss = 0.0
  for i, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播与优化
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
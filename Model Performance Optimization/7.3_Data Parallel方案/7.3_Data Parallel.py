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
# 开启 num_workers 加快数据加载速度
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2
)

# 2. 模型加载（使用现代 weights 参数替代已废弃的 pretrained=True）
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)  # 调整输出层用于 CIFAR-10

# 3. 使用 DataParallel 包装模型（多卡并行）
if torch.cuda.device_count() > 1:
  print(f"检测到 {torch.cuda.device_count()} 张 GPU，将使用 DataParallel 训练。")
  model = nn.DataParallel(model)
else:
  print("检测到单张 GPU 或 CPU 环境，按常规单机模式训练。")

# 将模型整体移动到 GPU
model = model.cuda()
model.train()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 4. 训练过程
num_epochs = 2
for epoch in range(num_epochs):
  running_loss = 0.0
  for i, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.cuda(), labels.cuda()

    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播与优化
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

  print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

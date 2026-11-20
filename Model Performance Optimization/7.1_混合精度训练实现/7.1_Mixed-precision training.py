import torch
import torch.nn as nn
import torch.optim as optim

# 现代 PyTorch 推荐的 AMP 导入方式
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# 数据预处理
transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

# 使用 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 使用预训练的 ResNet 模型（现代推荐 weights 参数）
model = models.resnet18(weights=ResNet18_Weights.DEFAULT).cuda()
model.train()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 混合精度训练需要使用 GradScaler
scaler = GradScaler("cuda")

# 混合精度训练循环
for epoch in range(1):
  running_loss = 0.0
  for inputs, labels in train_loader:
    inputs, labels = inputs.cuda(), labels.cuda()

    optimizer.zero_grad()

    # 显式指定 device_type
    with autocast(device_type="cuda"):
      outputs = model(inputs)
      loss = criterion(outputs, labels)

    # 使用缩放器管理梯度缩放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    running_loss += loss.item()

  print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

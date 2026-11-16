import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# CIFAR-10数据集加载
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 模型定义（使用ResNet18）
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 推荐使用 weights 替代废弃的 pretrained
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.cuda()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 梯度累积参数
accumulation_steps = 4  # 累积4步再更新

# 训练循环
num_epochs = 2
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    optimizer.zero_grad()  # 【修复 1】在每个 Epoch 开始时清空一次梯度

    total_batches = len(train_loader)

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播，梯度累积（损失缩放）
        loss = loss / accumulation_steps
        loss.backward()

        running_loss += loss.item() * accumulation_steps  # 还原真实 loss 用于打印

        # 【修复 2】判断触发更新的条件：
        # 条件 A：达到了指定的累积步数
        # 条件 B：已经是当前 DataLoader 的最后一个 Batch（防止尾部数据丢失）
        is_accumulation_step = (i + 1) % accumulation_steps == 0
        is_last_batch = (i + 1) == total_batches

        if is_accumulation_step or is_last_batch:
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清空梯度，开启下一轮累积

    print(f"Epoch {epoch + 1}, Loss: {running_loss / total_batches}")
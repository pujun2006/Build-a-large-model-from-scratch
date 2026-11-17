import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):

  def __init__(self, input_size, num_classes):
    super().__init__()
    self.fc = nn.Linear(input_size, num_classes)

  def forward(self, x):
    return self.fc(x)


def train_with_accumulation(accumulation_steps, num_epochs=10):
  # 1. 严格固定随机种子，确保每次实验初始权重和数据顺序完全一致
  torch.manual_seed(42)

  input_size, num_classes, num_samples = 10, 2, 1000
  # 2. 构造有真实规律的数据（线性可分+轻微噪声），而非纯随机噪声
  true_weights = torch.randn(input_size, 1)
  X = torch.randn(num_samples, input_size)
  logits = X @ true_weights + 0.5 * torch.randn(num_samples, 1)
  y = (logits.squeeze() > 0).long()

  dataset = TensorDataset(X, y)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  model = SimpleModel(input_size, num_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.01)

  print(
      f"\n--- 实验：accumulation_steps = {accumulation_steps} (等效 Batch Size ="
      f" {32 * accumulation_steps}) ---"
  )

  for epoch in range(num_epochs):
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(dataloader):
      outputs = model(inputs)
      raw_loss = criterion(outputs, labels)

      # 3. 累加真实未缩放的 loss 用于准确统计
      running_loss += raw_loss.item()

      # 缩放 loss 进行反向传播
      loss = raw_loss / accumulation_steps
      loss.backward()

      if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # 4. 处理无法整除时的残余梯度
    if len(dataloader) % accumulation_steps != 0:
      optimizer.step()
      optimizer.zero_grad()

    print(
        f"Epoch [{epoch + 1:2d}/{num_epochs}], Real Loss:"
        f" {running_loss / len(dataloader):.4f}"
    )


# 依次对比不同累积步数的损失下降情况
for steps in [1, 2, 4, 8]:
  train_with_accumulation(accumulation_steps=steps, num_epochs=10)
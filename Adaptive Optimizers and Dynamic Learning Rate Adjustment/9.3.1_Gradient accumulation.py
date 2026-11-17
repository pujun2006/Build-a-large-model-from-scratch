import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 设置随机种子以确保结果一致
torch.manual_seed(0)

# 2. 创建示例数据
# data_size=1000, batch_size=16 -> 总共有 63 个 Batch (63 % 4 = 3)
data_size, input_dim, output_dim, batch_size = 1000, 20, 2, 16
x = torch.randn(data_size, input_dim)
y = torch.randint(0, output_dim, (data_size,))
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
total_batches = len(dataloader)


# 3. 定义简单的线性模型
class SimpleModel(nn.Module):

  def __init__(self, input_dim, output_dim):
    super(SimpleModel, self).__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.fc(x)


model = SimpleModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 4. 设置梯度累积步数
accumulation_steps = 4  # 模拟 4 倍批量大小

# 5. 开始训练循环
model.train()
for epoch in range(2):  # 训练 2 个 epoch
  running_loss = 0.0
  optimizer.zero_grad()  # 每个 Epoch 开始前初始化梯度

  for i, (inputs, labels) in enumerate(dataloader):
    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps  # 关键：损失均分

    # 反向传播（梯度会自动累加到参数的 .grad 属性中）
    loss.backward()

    # 达到指定的累积步数时，执行参数更新
    if (i + 1) % accumulation_steps == 0:
      optimizer.step()  # 更新参数
      optimizer.zero_grad()  # 清空累积的梯度

    # 记录损失用于打印
    running_loss += loss.item() * accumulation_steps
    if (i + 1) % (accumulation_steps * 5) == 0:
      print(
          f"Epoch [{epoch+1}/2], Step [{i+1}/{total_batches}], Loss:"
          f" {running_loss / (accumulation_steps * 5):.4f}"
      )
      running_loss = 0.0

  # 【关键修复】处理每个 Epoch 结尾处不足 accumulation_steps 的残余梯度
  # 如果总 Batch 数不能被累积步数整除，这里进行兜底更新，避免数据白白浪费
  if total_batches % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()
    print(
        f"--- Epoch {epoch+1} 结束：已完成残余批次（{total_batches % accumulation_steps}个）的兜底梯度更新 ---"
    )

print("\n训练完成！")
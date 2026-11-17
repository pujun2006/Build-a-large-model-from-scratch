import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler  # 使用 PyTorch 2.0+ 推荐的公开基类


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()
# 优化器的初始 lr 可以直接设为你期望的目标峰值学习率
optimizer = optim.Adam(model.parameters(), lr=0.01)


class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, peak_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # 使用 self.last_epoch 代替 _step_count，这是官方更推荐的步数追踪方式
        # 注意：PyTorch 的 last_epoch 从 -1 开始，第一步执行 step() 后变为 0
        current_step = self.last_epoch + 1

        if current_step < self.warmup_steps:
            # 线性 Warmup
            return [self.peak_lr * (current_step / self.warmup_steps) for _ in self.base_lrs]
        else:
            # 保持峰值学习率不变
            return [self.peak_lr for _ in self.base_lrs]


# 设置超参数
warmup_steps = 100
peak_lr = 0.01  # 峰值学习率
scheduler = WarmupScheduler(optimizer, warmup_steps=warmup_steps, peak_lr=peak_lr)

# 模拟训练循环
data = torch.randn(32, 10)
target = torch.randn(32, 1)
criterion = nn.MSELoss()

for epoch in range(5):
    for batch in range(50):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 更新学习率

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {epoch + 1}, Batch: {batch + 1}, Learning Rate: {current_lr:.6f}, Loss: {loss.item():.6f}")
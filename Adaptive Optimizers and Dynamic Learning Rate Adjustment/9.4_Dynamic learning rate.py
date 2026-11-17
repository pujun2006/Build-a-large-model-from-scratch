import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 使用 PyTorch 官方的线性衰减调度器
# start_factor: 初始学习率相对于 base_lr 的倍数 (0.1 / 0.1 = 1.0)
# end_factor: 最终学习率相对于 base_lr 的倍数 (0.01 / 0.1 = 0.1)
# total_iters: 总共衰减的步数
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=20)

total_epochs = 20
lr_history = []

for epoch in range(total_epochs):
    model.train()
    inputs = torch.randn(5, 10)
    labels = torch.randn(5, 1)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, labels)
    loss.backward()
    optimizer.step()

    # 在每个 epoch（或 step）结束后调用 scheduler.step()
    lr_current = optimizer.param_groups[0]['lr']
    lr_history.append(lr_current)
    scheduler.step()

    print(f"Epoch {epoch + 1}/{total_epochs}, Learning Rate: {lr_current:.6f}, Loss: {loss.item():.6f}")
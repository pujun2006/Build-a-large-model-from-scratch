import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts
)


# 1. 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 2. 模拟数据集与损失函数
torch.manual_seed(42)
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)
criterion = nn.MSELoss()


# 3. 调度器工厂函数：通过字符串轻松切换不同的学习率策略
def get_scheduler(optimizer, scheduler_type):
    if scheduler_type == "StepLR":
        # 每 10 个 epoch 将学习率乘以 0.1
        return StepLR(optimizer, step_size=10, gamma=0.1)

    elif scheduler_type == "ExponentialLR":
        # 每个 epoch 学习率衰减为原来的 95%
        return ExponentialLR(optimizer, gamma=0.95)

    elif scheduler_type == "ReduceLROnPlateau":
        # 当验证/训练 loss 连续 5 个 epoch 不下降时，学习率乘以 0.5
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    elif scheduler_type == "CosineAnnealingLR":
        # 标准余弦退火，T_max 为半个周期的长度（这里设为总 epoch 数 50）
        return CosineAnnealingLR(optimizer, T_max=50)

    elif scheduler_type == "CosineAnnealingWarmRestarts":
        # 带热重启的余弦退火：T_0 是首个周期的 epoch 数，T_mult 是周期放大倍数
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ==========================================
# 4. 训练与测试主程序
# ==========================================
def train_with_scheduler(scheduler_name, num_epochs=50):
    print(f"\n========================================")
    print(f"开始运行学习率调度器: [{scheduler_name}]")
    print(f"========================================")

    # 初始化模型与优化器
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 获取对应的调度器
    scheduler = get_scheduler(optimizer, scheduler_name)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播与计算损失
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 获取当前生效的学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 【特别注意】ReduceLROnPlateau 需要传入监控的指标（如 loss），其他调度器直接调用 step()
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(loss.item())
        else:
            scheduler.step()

        # 每隔 10 个 epoch 打印一次日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:2d}/{num_epochs} | Loss: {loss.item():.4f} | Learning Rate: {current_lr:.6f}")


# 5. 测试不同的调度器策略
if __name__ == "__main__":
    # 可以依次测试上面列出的 5 种调度器
    schedulers_to_test = [
        "StepLR",
        "ExponentialLR",
        "ReduceLROnPlateau",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts"
    ]

    for name in schedulers_to_test:
        train_with_scheduler(name, num_epochs=50)

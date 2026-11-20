import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# 1. 准备模型与优化器
model = nn.Linear(10, 1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 2. 定义超参数
total_steps = 1000
warmup_steps = 100
max_lr = 1e-3
min_lr = 1e-4


# 3. 自定义 Warmup + 余弦退火的 Lambda 函数
def lr_lambda(current_step):
    if current_step < warmup_steps:
        # Warmup 阶段：线性上升 (从 0 线性增长到 1.0 倍的 base_lr)
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # 余弦退火阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))

        # 【修正】加上了缺失的乘号 *
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

        # 将归一化的余弦值映射到 [min_lr, max_lr] 对应的乘数区间
        lr_range = (max_lr - min_lr) / max_lr
        return min_lr / max_lr + lr_range * cosine_decay


# 4. 初始化调度器
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# 5. 模拟训练循环
for step in range(total_steps):
    # 注：实际训练中，此处应有前向传播、计算 loss、反向传播
    optimizer.step()
    scheduler.step()  # 更新学习率

    # 打印当前学习率
    if step % 200 == 0:
        print(f"Step {step}, LR: {optimizer.param_groups[0]['lr']:.6f}")

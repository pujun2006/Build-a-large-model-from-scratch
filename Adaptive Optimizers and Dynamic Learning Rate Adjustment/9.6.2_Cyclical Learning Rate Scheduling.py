import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import matplotlib
matplotlib.use('TkAgg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt

# 1. 构建一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 初始化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. 设置循环学习率调度器 (CyclicLR)
# 适当调大 step_size_up，让学习率上升和下降的周期更加平缓、易于观察
scheduler = CyclicLR(
    optimizer,
    base_lr=0.001,    # 最小学习率 (下界)
    max_lr=0.01,      # 最大学习率 (上界)
    step_size_up=50,  # 学习率从 base_lr 上升到 max_lr 所需的步数 (Step)
    mode='triangular' # 循环模式：标准三角波
)

# 4. 创建模拟训练数据
# 设定每个 Epoch 包含 50 个 Batch
data = torch.randn(50, 10)
target = torch.randn(50, 1)

# 用于记录每个 step 的学习率变化
lr_list = []

# 5. 模拟训练过程
num_epochs = 20  # 总共训练 20 个 Epoch (总计 20 * 50 = 1000 个 Step)
for epoch in range(num_epochs):
    for batch in range(len(data)):
        inputs = data[batch].unsqueeze(0)  # 单个样本输入
        labels = target[batch].unsqueeze(0)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新学习率 (必须紧跟在 optimizer.step 之后)
        scheduler.step()

        # 记录当前学习率
        lr_list.append(optimizer.param_groups[0]['lr'])

# 6. 可视化学习率变化情况
plt.figure(figsize=(10, 5))
plt.plot(lr_list, label="Cyclic Learning Rate", color="blue")
plt.xlabel("Training Step", fontsize=12)
plt.ylabel("Learning Rate", fontsize=12)
plt.title("Cyclic Learning Rate Schedule (Triangular Mode)", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
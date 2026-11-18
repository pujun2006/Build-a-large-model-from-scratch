import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune


# ==========================================
# 1. 定义模型、损失函数与优化器
# ==========================================
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# ==========================================
# 2. 模拟模型训练（Pre-training）
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# 生成一些虚拟训练数据
dummy_train_loader = [
    (torch.randn(32, 784), torch.randint(0, 10, (32,)))
    for _ in range(10)
]

print("开始预训练模型...")
for epoch in range(2):
    train_epoch(model, dummy_train_loader, criterion, optimizer)
print("预训练完成！\n")


# ==========================================
# 3. 辅助函数：统计权重稀疏度（0 的比例）
# ==========================================
def check_sparsity(model):
    total_weights = 0
    zero_weights = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_weights += module.weight.nelement()
            zero_weights += (module.weight == 0).sum().item()
    print(f"总权重数: {total_weights}")
    print(f"其中为 0 的权重数: {zero_weights}")
    print(f"当前稀疏度 (Sparsity): {zero_weights / total_weights * 100:.2f}%\n")


print("--- 剪枝前的稀疏度 ---")
check_sparsity(model)

# ==========================================
# 4. 应用权重剪枝（L1 非结构化剪枝）
# ==========================================
print("正在应用 L1 非结构化权重剪枝（剪掉权重绝对值最小的 50%）...")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # 对全连接层的 weight 应用基于 L1 范数的剪枝，剪枝比例为 50%
        prune.l1_unstructured(module, name='weight', amount=0.5)

print("--- 剪枝后（未微调）的稀疏度 ---")
check_sparsity(model)

# ==========================================
# 5. 微调阶段（Fine-tuning / Retraining）
# ==========================================
# 提示：剪枝后模型的精度通常会下降，必须通过几轮微调让剩下的权重适应稀疏结构
print("开始进行剪枝后的微调（Fine-tuning）...")
fine_tune_optimizer = optim.SGD(model.parameters(), lr=0.001)  # 微调时通常用更小的学习率
for epoch in range(3):
    train_epoch(model, dummy_train_loader, criterion, fine_tune_optimizer)
print("微调完成！\n")

# ==========================================
# 6. 固化剪枝结果（Remove Pruning Mask）
# ==========================================
# 当微调结束后，如果不再需要动态调整掩码，可以调用 prune.remove
# 这会将 mask 永久乘以 weight，并将 weight 转为普通的张量，释放显存与消除额外开销
print("正在固化剪枝结果（永久移除 Mask 缓冲区）...")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.remove(module, 'weight')

print("--- 固化后的最终模型状态 ---")
check_sparsity(model)

# 此时模型的权重矩阵中已经有 50% 的元素物理数值为 0
print(f"fc1 层的权重矩阵形状: {model.fc1.weight.shape}")
print(f"fc1 层权重前 3x3 矩阵:\n {model.fc1.weight[:3, :3]} ...")
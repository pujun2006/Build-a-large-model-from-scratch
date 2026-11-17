# Step 1: 导入所需库
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib
matplotlib.use('TkAgg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt


# Step 2: 定义LAMB优化器
class LAMB(optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(LAMB, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # 初始化状态字典
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # 更新一阶、二阶动量
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算自适应学习率
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                r1 = p.data.pow(2).sum().sqrt()
                r2 = exp_avg.pow(2).sum().sqrt()
                if r1 != 0 and r2 != 0:
                    trust_ratio = r1 / r2
                else:
                    trust_ratio = 1.0
                step_size = step_size * trust_ratio

                # 权重衰减项
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                # 更新参数
                p.data.add_(exp_avg, alpha=-step_size)
        return loss


# Step 3: 定义一个简单的模型和数据集
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 生成数据
torch.manual_seed(42)
x_train = torch.linspace(0, 10, 100).view(-1, 1)
y_train = 3.0 * x_train + 2.0 + torch.randn(x_train.size()) * 2.0

# Step 4: 初始化模型、LAMB优化器和损失函数
model = SimpleLinearModel()
optimizer = LAMB(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Step 5: 训练模型
losses = []
num_epochs = 200

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    # 后向传播和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 6: 绘制损失值
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss with LAMB Optimizer')
plt.show()

# Step 7: 输出模型的参数
print("\n模型的权重和偏置：")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")
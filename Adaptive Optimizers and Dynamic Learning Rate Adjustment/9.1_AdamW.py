# Step 1: 导入所需的库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import matplotlib
matplotlib.use('TkAgg')  # 必须在导入 pyplot 之前设置
import matplotlib.pyplot as plt  # 导入Matplotlib进行绘图

# Step 2: 定义一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()  # 初始化父类
        self.linear = nn.Linear(1, 1)  # 定义一个线性层，输入和输出维度均为1

    def forward(self, x):
        return self.linear(x)  # 前向传播：计算输入x的线性输出

# Step 3: 生成一些示例数据（输入x_train和目标输出y_train）
torch.manual_seed(0)  # 设置随机种子，保证结果可重复
x_train = torch.linspace(0, 10, 100).view(-1, 1)  # 生成100个均匀分布的点作为输入数据
y_train = 2 * x_train + 3 + torch.randn(x_train.size()) * 2  # 已修正：加上乘号并去掉多余的连字符

# Step 4: 初始化模型
model = LinearRegressionModel()  # 创建模型实例

# Step 5: 定义AdamW优化器
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)  # 学习率为0.01，权重衰减为0.01

# Step 6: 定义损失函数（均方误差损失）
criterion = nn.MSELoss()  # 使用均方误差（MSE）作为损失函数

# Step 7: 准备存储训练损失值的列表
losses = []  # 用于记录每个epoch的损失值

# Step 8: 训练模型
num_epochs = 200  # 设定训练的迭代次数

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_train)  # 使用模型计算预测值
    loss = criterion(y_pred, y_train)  # 计算预测值与真实值的损失

    # 记录损失值
    losses.append(loss.item())  # 将损失值添加到列表中

    # 后向传播与优化步骤
    optimizer.zero_grad()  # 清除上一步的梯度信息
    loss.backward()  # 计算当前损失对参数的梯度
    optimizer.step()  # 更新模型参数

    # 每隔20个epoch打印一次当前的损失值
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 9: 绘制训练损失曲线
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')  # 横轴标签
plt.ylabel('Loss')  # 纵轴标签
plt.title('Training Loss with AdamW Optimizer')  # 图的标题
plt.show()  # 显示图像

# Step 10: 输出模型的最终参数
print("\n模型参数（权重和偏置）：")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

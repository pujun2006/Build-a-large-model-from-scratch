# 导入所需库
import os
import torch
import torch.nn as nn
import torch.optim as optim


# 1. 定义简单的前馈神经网络模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 2. 训练主函数（支持定期保存检查点）
def train(model, criterion, optimizer, start_epoch, total_epochs, save_interval, checkpoint_dir='./checkpoints'):
    # 确保检查点目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"--> 开始训练，从第 {start_epoch + 1} 轮到第 {total_epochs} 轮...")

    for epoch in range(start_epoch, total_epochs):
        model.train()

        # 生成模拟随机数据 (Batch Size = 32, Input Dim = 10)
        inputs = torch.randn(32, 10)
        labels = torch.randn(32, 1)

        # 前向传播、计算损失
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和参数优化
        loss.backward()
        optimizer.step()

        current_epoch = epoch + 1

        # 定期保存检查点（包含模型权重、优化器动量状态及当前轮次）
        if current_epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{current_epoch}.pth")
            torch.save({
                'epoch': current_epoch,  # 保存当前已完成的 epoch
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"-> 已保存检查点: {checkpoint_path}")

        print(f"Epoch [{current_epoch}/{total_epochs}], Loss: {loss.item():.4f}")

    print("Training finished.")


# 3. 恢复检查点函数（断点续训核心）
def resume_training(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"正在从以下路径加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # 恢复模型权重与优化器状态（含 Adam 动量缓存）
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        saved_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"-> 检查点加载成功！将从第 {saved_epoch + 1} 轮开始继续训练（断点处 Loss: {loss:.4f}）")
        return saved_epoch
    else:
        print(f"未在指定路径找到检查点: {checkpoint_path}")
        return 0


if __name__ == "__main__":
    # 初始化超参数与组件
    input_size = 10
    hidden_size = 20
    output_size = 1

    model = SimpleModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    checkpoint_dir = './checkpoints'
    target_checkpoint = os.path.join(checkpoint_dir, "model_epoch_10.pth")

    # ==========================================
    # 场景模拟一：全新开始训练（或者你可以先运行一次让它保存出第 10 轮的权重）
    # ==========================================
    # 假设我们计划总共训练 20 个 Epoch，每 5 个 Epoch 保存一次检查点
    total_epochs = 20
    save_interval = 5

    # 检查是否存在历史检查点，如果存在则自动恢复，实现断点续训
    start_epoch = resume_training(target_checkpoint, model, optimizer)

    # 如果没有历史检查点（start_epoch = 0），则从头开始；如果加载了第 10 轮的检查点，则从第 10 轮无缝接续
    if start_epoch < total_epochs:
        train(model, criterion, optimizer, start_epoch=start_epoch, total_epochs=total_epochs,
              save_interval=save_interval, checkpoint_dir=checkpoint_dir)
    else:
        print("模型已经完成了所有训练轮次！")

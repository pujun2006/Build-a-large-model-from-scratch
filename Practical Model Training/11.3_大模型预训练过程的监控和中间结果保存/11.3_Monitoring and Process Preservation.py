# 导入所需库
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


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


# 2. 定义模拟数据集
class RandomDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 3. 训练主函数
def train():
    # 配置路径
    log_dir = "./runs/experiment"
    checkpoint_dir = "./checkpoints"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化超参数与组件
    input_size = 10
    hidden_size = 20
    output_size = 1
    epochs = 50
    batch_size = 32
    save_interval = 10  # 每 10 个 epoch 保存一次检查点

    model = SimpleModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 构造数据集与 DataLoader
    dataset = RandomDataset(num_samples=256, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 使用 TensorBoard 监控器（采用 try-finally 确保安全关闭）
    writer = SummaryWriter(log_dir)

    try:
        global_step = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1

                # 记录每个 Step 的 Loss 到 TensorBoard
                writer.add_scalar('Loss/train_step', loss.item(), global_step)

            # 计算并记录每个 Epoch 的平均 Loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch + 1)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

            # 定期保存检查点（包含模型权重、优化器状态及当前轮次）
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                }, checkpoint_path)
                print(f"-> 已保存检查点: {checkpoint_path}")
            # 【新增】每轮结束后强制将缓冲区数据写入磁盘，避免 0 KB
            writer.flush()
        print("Training complete.")

    finally:
        # 确保无论训练是否异常中断，TensorBoard 句柄都能正确关闭并落盘
        writer.close()


if __name__ == "__main__":
    train()

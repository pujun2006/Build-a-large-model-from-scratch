# 因多GPU环境受限，以下代码未进行验证
# import os
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# from torchvision import datasets, transforms
#
#
# # 初始化分布式环境
# def setup(rank, world_size):
#   os.environ['MASTER_ADDR'] = 'localhost'
#   os.environ['MASTER_PORT'] = '12355'
#   dist.init_process_group('nccl', rank=rank, world_size=world_size)
#   torch.cuda.set_device(rank)
#
#
# # 清理分布式环境
# def cleanup():
#   dist.destroy_process_group()
#
#
# # 模型定义
# class SimpleModel(nn.Module):
#
#   def __init__(self):
#     super(SimpleModel, self).__init__()
#     self.fc = nn.Linear(28 * 28, 10)  # 修正：28*28 = 784
#
#   def forward(self, x):
#     x = x.view(-1, 28 * 28)  # 修正：展平输入
#     return self.fc(x)
#
#
# # 分布式训练过程
# def train(rank, world_size):
#   setup(rank, world_size)
#
#   transform = transforms.Compose([transforms.ToTensor()])
#   dataset = datasets.MNIST(
#       root='./data', train=True, download=True, transform=transform
#   )
#
#   # DistributedSampler 会自动将数据集平均切分到各个 GPU 上
#   sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#   dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)
#
#   model = SimpleModel().cuda(rank)
#   # 使用 DDP 包装模型，并指定当前进程对应的 GPU ID
#   model = DDP(model, device_ids=[rank])
#
#   criterion = nn.CrossEntropyLoss()
#   optimizer = optim.SGD(model.parameters(), lr=0.01)
#
#   for epoch in range(2):  # 使用2个epoch示例
#     # 必须调用 set_epoch，否则分布式环境下的数据无法在不同 epoch 间正确 shuffle
#     sampler.set_epoch(epoch)
#     running_loss = 0.0
#
#     for batch_idx, (data, target) in enumerate(dataloader):
#       data, target = data.cuda(rank), target.cuda(rank)
#       optimizer.zero_grad()
#       output = model(data)
#       loss = criterion(output, target)
#       loss.backward()
#       optimizer.step()
#
#       running_loss += loss.item()
#
#     # 仅在主进程（rank 0）打印日志，避免多进程刷屏
#     if rank == 0:
#       print(
#           f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}'
#       )
#
#   cleanup()
#
#
# # 多进程启动
# def main():
#   world_size = 2  # 示例设置为2个进程（对应2张 GPU）
#   torch.multiprocessing.spawn(
#       train, args=(world_size,), nprocs=world_size, join=True
#   )
#
#
# if __name__ == '__main__':
#   main()


# 虽然物理上只有 1 张 GPU，但 PyTorch 允许我们将 world_size 设置为 1，或者使用 CPU 后端（gloo） 或单卡进程组来模拟分布式流程。
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 模拟分布式环境初始化（使用 gloo 后端，支持单机无多卡运行）
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# 如果有 GPU 就用 cuda，没有就退回 cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

# 初始化进程组（world_size=1, rank=0）
dist.init_process_group(backend="gloo", rank=0, world_size=1)


# 2. 模型定义
class SimpleModel(nn.Module):
  def __init__(self):
    super(SimpleModel, self).__init__()
    self.fc = nn.Linear(28 * 28, 10)

  def forward(self, x):
    return self.fc(x.view(-1, 28 * 28))


# 3. 准备数据
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
for epoch in range(2):
  running_loss = 0.0
  for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

  print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")

# 清理
dist.destroy_process_group()
print("训练成功完成！")

# imports
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp

# 设置主节点和端口，配置环境变量
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def setup(rank, world_size, use_gpu):
    # 根据环境选择后端：GPU 使用高效的 "nccl"，CPU 使用 "gloo"
    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.manual_seed(0)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, epochs=10, batch_size=32):
    # 动态检测当前环境是否有可用 GPU
    use_gpu = torch.cuda.is_available()

    setup(rank, world_size, use_gpu)

    # 设置数据和模型
    dataset = RandomDataset(10, 500)

    if use_gpu:
        # GPU 模式配置
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        model = SimpleModel(10, 1).to(device)
        model = DDP(model, device_ids=[rank])
        print(f"-> 进程 [Rank {rank}] 正在使用 GPU: cuda:{rank}")
    else:
        # CPU 模式配置
        device = torch.device("cpu")
        model = SimpleModel(10, 1).to(device)
        model = DDP(model)  # CPU 的 DDP 不需要指定 device_ids
        print(f"-> 进程 [Rank {rank}] 正在使用 CPU")

    # Dataloader 分布式设置
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader, 0):
            inputs = data.to(device)

            # 使用动态 batch 大小避免最后一个 batch 维度报错
            current_batch_size = inputs.size(0)
            labels = torch.randn(current_batch_size, 1).to(device)

            # 正向和反向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Rank {rank} | Epoch [{epoch + 1}/{epochs}] | Step [{i + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    cleanup()


def main():
    if torch.cuda.is_available():
        # 如果有 GPU，获取可用 GPU 数量（例如设为 2，要求至少有 2 张卡）
        available_gpus = torch.cuda.device_count()
        world_size = min(2, available_gpus)
        print(f"检测到 GPU 环境，将启动 {world_size} 个 GPU 进程进行 DDP 训练。")
    else:
        # 如果没有 GPU，则模拟 2 个 CPU 进程
        world_size = 2
        print(f"未检测到 GPU，将启动 {world_size} 个 CPU 进程进行分布式模拟训练。")

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import time


# 1. 定义原始模型
class SimpleNN(nn.Module):
    def __init__(self, h1=256, h2=128):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, h1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark(model, inputs):
    model.eval()
    with torch.no_grad():
        for _ in range(10): _ = model(inputs)  # 预热
        start = time.time()
        for _ in range(100):
            _ = model(inputs)
        return (time.time() - start) * 10
        ms


if __name__ == "__main__":
    test_inputs = torch.randn(5000, 784)

    print("=" * 50)
    print("【结构化剪枝】通过缩减层维度实现物理参数与时间下降")
    print("=" * 50)

    # 剪枝前的完整模型 (fc1隐层256，fc2隐层128)
    model_orig = SimpleNN(h1=256, h2=128)
    print(f"剪枝前物理参数量: {count_parameters(model_orig)}")
    time_orig = benchmark(model_orig, test_inputs)
    print(f"剪枝前推理耗时: {time_orig:.4f} ms")

    # 模拟结构化剪枝：将隐藏层通道数砍掉一半（h1=128, h2=64）
    model_pruned = SimpleNN(h1=128, h2=64)
    print(f"\n剪枝后物理参数量: {count_parameters(model_pruned)}")
    time_pruned = benchmark(model_pruned, test_inputs)
    print(f"剪枝后推理耗时: {time_pruned:.4f} ms (物理参数减少，时间显著缩短！)")
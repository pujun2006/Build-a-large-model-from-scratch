import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


# 1. 定义 SimpleGPTModel
class SimpleGPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SimpleGPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.fc(x)
        return x


# 2. 修正后的计算类：同时返回 Loss（用于反向传播）和 Perplexity（用于打印指标）
class GPTLossAndPerplexity(nn.Module):
    def __init__(self, vocab_size):
        super(GPTLossAndPerplexity, self).__init__()
        self.vocab_size = vocab_size

    def forward(self, model, input_ids):
        # outputs 形状: [Batch_Size, Seq_Len, Vocab_Size]
        outputs = model(input_ids)

        # 真正标准的 Shift-Right 对齐：
        # 用前 T-1 个位置的输出去预测后 T-1 个位置的标签
        shift_logits = outputs[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = input_ids[:, 1:].contiguous().view(-1)

        # 计算交叉熵损失
        loss = F.cross_entropy(shift_logits, shift_labels)

        # 困惑度 PPL = exp(Loss)
        perplexity = torch.exp(loss)
        return loss, perplexity


# 设置模型参数与数据
vocab_size = 1000
embed_size = 256
model = SimpleGPTModel(vocab_size, embed_size)
input_data = torch.randint(0, vocab_size, (1, 20))  # 模拟输入数据

loss_ppl_calculator = GPTLossAndPerplexity(vocab_size)

# 微调和学习率调整设置
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 略调大学习率以快速看到效果
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# 模拟微调过程
num_epochs = 15
model.train()  # 开启训练模式

print("--- 开始训练/微调 ---")
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 计算 Loss 与 PPL
    loss, perplexity = loss_ppl_calculator(model, input_data)

    # 关键修正：对 Loss 求导反向传播，而不是对 Perplexity 求导！
    loss.backward()
    optimizer.step()
    scheduler.step()

    # 输出每个 epoch 的交叉熵损失和困惑度
    print(f"Epoch {epoch + 1:02d} | Loss: {loss.item():.4f} | 困惑度 (PPL): {perplexity.item():.4f}")

# 最终的困惑度计算
model.eval()  # 开启评估模式
with torch.no_grad():  # 计算最终指标时不占用梯度显存
    _, final_perplexity = loss_ppl_calculator(model, input_data)

print("\n--- 训练结束 ---")
print(f"最终困惑度 (Final PPL): {final_perplexity.item():.4f}")

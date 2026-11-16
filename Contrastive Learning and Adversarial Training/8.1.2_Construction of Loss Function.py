import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # 计算正样本对之间的欧几里得距离
        pos_distance = F.pairwise_distance(anchor, positive, keepdim=True)

        # 计算负样本对之间的欧几里得距离
        neg_distance = F.pairwise_distance(anchor, negative, keepdim=True)

        # 损失函数包含两部分：正样本对距离的平方，以及负样本对距离和 margin 之差的平方
        loss_positive = torch.mean(pos_distance ** 2)
        loss_negative = torch.mean(F.relu(self.margin - neg_distance) ** 2)

        # 总的对比损失是正负样本对损失的加权和
        loss = loss_positive + loss_negative
        return loss


# 创建损失函数实例
margin = 1.0  # margin 值可以根据具体任务进行调整
contrastive_loss = ContrastiveLoss(margin=margin)

# 示例数据
# anchor、positive、negative 代表一个批次的锚点样本、正样本对和负样本对的嵌入向量
# 使用随机数据生成三组向量以模拟嵌入结果
anchor = torch.randn(10, 128)  # 10 个样本，每个样本的嵌入向量大小为 128
positive = torch.randn(10, 128)  # 正样本对
negative = torch.randn(10, 128)  # 负样本对

# 计算损失
loss = contrastive_loss(anchor, positive, negative)

print(f"Contrastive Loss: {loss.item()}")

# 正样本对距离和负样本对距离
pos_distance = F.pairwise_distance(anchor, positive, keepdim=True)
neg_distance = F.pairwise_distance(anchor, negative, keepdim=True)

print(f"Positive Pair Distance (mean): {pos_distance.mean().item()}")
print(f"Negative Pair Distance (mean): {neg_distance.mean().item()}")

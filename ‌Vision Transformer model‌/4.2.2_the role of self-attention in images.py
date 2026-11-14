import torch
import torch.nn as nn
import numpy as np


# 定义自注意力机制模块（已修复线性投影逻辑）
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size必须是heads数量的整数倍"

        # 修正：将整体验证或单头投影改为映射整个 embed_size
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 1. 先通过线性层投影，再分割多头 [N, seq_len, embed_size] -> [N, seq_len, heads, head_dim]
        v = self.value(values).reshape(N, value_len, self.heads, self.head_dim)
        k = self.key(keys).reshape(N, key_len, self.heads, self.head_dim)
        q = self.query(query).reshape(N, query_len, self.heads, self.head_dim)

        # 2. 自注意力计算
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k]) / (self.head_dim ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        # 3. 聚合值并通过输出线性层
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(
            N, query_len, self.embed_size
        )

        return self.fc_out(out)


# Transformer编码器块
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 遵循原论文，使用 GELU 激活函数
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # 现代 Pre-LN 结构更稳定，此处按原代码的 Post-LN 逻辑保留或可自行改为 Pre-LN
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


# Vision Transformer模型定义
class VisionTransformerWithAttention(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_size=768, num_heads=8, num_layers=6,
                 forward_expansion=4, dropout=0.1, num_classes=1000):
        super(VisionTransformerWithAttention, self).__init__()
        self.patch_embedding = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_size))

        self.transformer = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings

        for layer in self.transformer:
            x = layer(x, x, x, mask=None)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


# 测试代码
if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    embed_size = 768
    num_heads = 8
    num_layers = 6
    num_classes = 10

    x = torch.randn(1, 3, img_size, img_size)
    model = VisionTransformerWithAttention(img_size=img_size, patch_size=patch_size, embed_size=embed_size,
                                           num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)
    output = model(x)

    print("输出形状:", output.shape)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 1. 图像分块和嵌入模块 (Patch Embedding)
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 使用 stride=patch_size 的卷积将图像切块并映射为 embed_size 向量
        self.projection = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))

    def forward(self, x):
        # [batch_size, embed_size, H/patch, W/patch]
        x = self.projection(x)
        x = x.flatten(2)  # [batch_size, embed_size, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_size]

        # 拼接分类标识符 ([CLS] Token)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_size]

        # 加上可学习的位置编码
        x = x + self.position_embeddings
        return x


# 2. Transformer 编码器块 (标准 Pre-LN 结构)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        # 修正 1：设置 batch_first=True 以匹配 [batch, seq_len, embed_size] 输入
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 修正 2：符合 ViT 原论文，使用 GELU 替换 ReLU
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 标准 Pre-LN Residual Block 结构
        norm_x = self.norm1(x)
        attention_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + self.dropout(attention_out)

        feed_forward_out = self.feed_forward(self.norm2(x))
        x = x + feed_forward_out
        return x


# 3. Vision Transformer 主模型
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_size=768, num_heads=8, num_layers=6,
                 forward_expansion=4, dropout=0.1, num_classes=1000):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size, img_size)

        self.transformer = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        for layer in self.transformer:
            x = layer(x)

        # 提取 [CLS] Token 对应输出进行分类预测
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)


# 示例数据生成
def generate_sample_image(img_size=224):
    img = np.random.rand(img_size, img_size, 3)
    # plt.imshow(img)
    # plt.title("Sample Image")
    # plt.show() # 注释掉防止阻塞执行
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()


# 运行测试
img = generate_sample_image()
model = VisionTransformer()
output = model(img)

print("运行成功！")
print("输出张量形状:", output.shape)  # 期望输出: torch.Size([1, 1000])s

import torch
import torch.nn as nn
import numpy as np


# ==========================================
# 1. 图像分块与嵌入模块 (Patch Embedding)
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用 stride=patch_size 的二维卷积将图像切块并映射到 embed_size 维度
        self.projection = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)

        # 可学习的 [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

        # 可学习的位置编码 (包含 [CLS] Token，所以长度为 num_patches + 1)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.shape[0]

        # [B, C, H, W] -> [B, embed_size, H/P, W/P]
        x = self.projection(x)

        # 展平并转置: [B, embed_size, num_patches] -> [B, num_patches, embed_size]
        x = x.flatten(2).transpose(1, 2)

        # 扩展并拼接 [CLS] Token: [B, num_patches + 1, embed_size]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 加上可学习的位置编码并经过 Dropout
        x = self.dropout(x + self.position_embeddings)
        return x


# ==========================================
# 2. Transformer 编码器块 (标准 Pre-LN 结构)
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()

        # 关键修正：设置 batch_first=True 以匹配 [batch, seq_len, embed_size]
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 严格遵循原论文，使用 GELU 激活函数
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. 现代 Pre-LN 多头自注意力 + 残差连接
        norm_x = self.norm1(x)
        attention_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + self.dropout(attention_out)

        # 2. 现代 Pre-LN 前馈网络 + 残差连接
        feed_forward_out = self.feed_forward(self.norm2(x))
        x = x + feed_forward_out

        return x


# ==========================================
# 3. Vision Transformer 主模型
# ==========================================
class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_size=768,
            num_heads=8,
            num_layers=6,
            forward_expansion=4,
            dropout=0.1,
            num_classes=10
    ):
        super(VisionTransformer, self).__init__()

        # 图像分块与嵌入层
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size, img_size)

        # 堆叠多层 Transformer Encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

        # 分类预测头 (MLP Head)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        # 图像转 Token 序列: [B, 3, 224, 224] -> [B, 197, 768]
        x = self.patch_embedding(x)

        # 经过多个 Transformer 编码器块
        for layer in self.transformer:
            x = layer(x)

        # 提取 [CLS] Token 对应的特征进行分类 (索引为 0)
        cls_token_final = x[:, 0]

        # 输出分类 Logits
        return self.mlp_head(cls_token_final)


# ==========================================
# 4. 测试与运行验证
# ==========================================
if __name__ == "__main__":
    # 超参数设置
    IMG_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    EMBED_SIZE = 768
    NUM_HEADS = 8
    NUM_LAYERS = 6
    NUM_CLASSES = 10
    BATCH_SIZE = 2  # 测试多样本 Batch 维度

    # 模拟输入一个 Batch 的图像数据 (例如 2 张 224x224 的 RGB 图像)
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)

    # 实例化 Vision Transformer 模型
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    )

    # 设置为评估模式
    model.eval()

    with torch.no_grad():
        output = model(x)

    print("=== ViT 模型运行成功！ ===")
    print(f"输入图像张量形状: {x.shape}")
    print(f"模型输出张量形状: {output.shape} (预期: torch.Size([{BATCH_SIZE}, {NUM_CLASSES}]))")
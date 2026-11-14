import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


# 1. 图像分块和嵌入模块 (Patch Embedding)
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 使用 stride=patch_size 的卷积切块并投影
        self.projection = nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_size))

    def forward(self, x):
        x = self.projection(x)  # [B, embed_size, H/P, W/P]
        x = x.flatten(2)  # [B, embed_size, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_size]

        # 拼接 [CLS] Token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_size]

        # 加上可学习的位置编码
        x = x + self.position_embeddings
        return x


# 2. Transformer 编码器块 (修正为标准的 Pre-LN 结构)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        # 【关键修正 1】：设置 batch_first=True 以匹配 [batch, seq_len, embed_size]
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 【关键修正 2】：使用 GELU 替换 ReLU，并加入 Dropout
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 【关键修正 3】：标准 Pre-LN 结构 (先 Norm 再计算，最后残差相加)
        norm_x = self.norm1(x)
        attention_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + self.dropout(attention_out)

        feed_forward_out = self.feed_forward(self.norm2(x))
        x = x + feed_forward_out
        return x


# 3. Vision Transformer 主模型
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_size=768, num_heads=8, num_layers=6,
                 forward_expansion=4, dropout=0.1, num_classes=10):
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

        # 提取 [CLS] Token 并预测
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)


# --- 测试运行 ---
# 1. 模拟或加载图像
try:
    img_path = "cat.jpeg"
    image = Image.open(img_path).convert("RGB")
except FileNotFoundError:
    # 如果本地没有 test.jpg，则自动生成一张随机虚幻图片测试，避免报错
    image = Image.fromarray((torch.rand(224, 224, 3).numpy() * 255).astype('uint8'))

# 2. 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 建议加上归一化
])

image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# 3. 初始化 ViT 并进行前向传播
model = VisionTransformer(num_classes=10)
model.eval()  # 切换到评估模式

with torch.no_grad():
    output = model(image_tensor)

print("运行成功！")
print("模型输出张量:", output)
print("输出形状:", output.shape)  # 期望输出: torch.Size([1, 10])
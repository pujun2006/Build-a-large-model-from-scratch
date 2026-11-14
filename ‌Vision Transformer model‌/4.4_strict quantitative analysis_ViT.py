import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
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


# 2. Transformer 编码器块 (修正为标准的 Pre-LN 结构，并安全返回 Attention Weights)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        # 【关键修正 1】：显式设置 batch_first=True，完美适配 [Batch, Seq_Len, Embed_Dim]
        self.attention = nn.MultiheadAttention(embed_size, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 【关键优化】：使用 GELU 替换 ReLU，并加入 Dropout
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 【关键修正 2】：标准 Pre-LN 结构 (先 Norm 再计算自注意力，最后残差相加)
        norm_x = self.norm1(x)

        # 开启 average_attn_weights=True，使返回的 weights 维度干净地保持为 [Batch, Seq_Len, Seq_Len]
        attention_out, weights = self.attention(
            norm_x, norm_x, norm_x,
            need_weights=True,
            average_attn_weights=True
        )
        x = x + self.dropout(attention_out)

        # 第二个 Pre-LN 与前馈网络
        feed_forward_out = self.feed_forward(self.norm2(x))
        x = x + feed_forward_out

        return x, weights


# 3. Vision Transformer 主模型（带注意力权重返回）
class VisionTransformerWithAttention(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_size=768,
                 num_heads=8, num_layers=6, forward_expansion=4, dropout=0.1, num_classes=10):
        super(VisionTransformerWithAttention, self).__init__()
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
        weights_list = []

        for layer in self.transformer:
            x, weights = layer(x)
            weights_list.append(weights)

        # 提取 [CLS] Token (位于序列索引 0 处) 进行最终分类
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final), weights_list


# --- 4. 严格量化分析函数 (规避危险的盲目 squeeze) ---
def quantify_attention(attention_weights):
    layer_analysis = []
    print("====== 注意力权重严格量化分析 ======")
    for layer_idx, weights in enumerate(attention_weights):
        # weights 形状: [Batch, Seq_Len, Seq_Len]
        weights_np = weights.cpu().detach().numpy()

        # 沿着 Batch 维度计算统计量（如果 Batch=1，则等价于当前样本的结果）
        mean_weights = np.mean(weights_np, axis=0)
        var_weights = np.var(weights_np, axis=0)

        layer_summary = {
            "layer": layer_idx + 1,
            "mean_attention": mean_weights,
            "variance_attention": var_weights
        }
        layer_analysis.append(layer_summary)

        print(f"Layer {layer_idx + 1} -> 权重矩阵形状: {weights_np.shape} | 均值矩阵形状: {mean_weights.shape}")

    return layer_analysis


# --- 5. 测试与运行 ---
if __name__ == "__main__":
    # 加载本地图像或自动生成模拟图像
    img_path = "test.jpg"
    try:
        image = Image.open(img_path).convert("RGB")
        print(f"成功加载本地图像: {img_path}")
    except FileNotFoundError:
        print(f"未找到本地图像 '{img_path}'，正在自动生成一张随机测试图像...")
        image = Image.fromarray((torch.rand(224, 224, 3).numpy() * 255).astype('uint8'))

    # 图像预处理流水线
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # 扩展 Batch 维度 -> [1, 3, 224, 224]

    # 初始化修正后的 ViT 模型
    model = VisionTransformerWithAttention(num_classes=10)
    model.eval()  # 切换到评估模式

    # 前向传播并获取输出与注意力权重
    with torch.no_grad():
        output, attention_weights = model(image_tensor)

    print("\n运行成功！")
    print("模型输出张量形状:", output.shape)  # 期望: torch.Size([1, 10])
    print(f"总共收集到 {len(attention_weights)} 层的注意力权重。\n")

    # 执行严格量化分析
    layer_analysis = quantify_attention(attention_weights)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np  # 确保导入了 numpy

# 数据准备
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# 加载CIFAR-10数据集
dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 模型定义（特征提取编码器 + 分类头）
class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128)
        )

    def forward(self, x):
        return self.conv(x)

# 分类模型
class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)

# 训练和验证方法
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy

# 初始化与训练
encoder = FeatureEncoder()
classification_model = ClassificationModel(encoder)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classification_model.parameters(), lr=1e-3)

train_model(classification_model, train_loader, criterion, optimizer, num_epochs=5)
validate_model(classification_model, val_loader, criterion)

# 使用t-SNE可视化编码器输出的嵌入向量分布
def visualize_embeddings(encoder, loader):
    encoder.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, label in loader:
            emb = encoder(images).detach().cpu().numpy()
            embeddings.extend(emb)
            labels.extend(label.numpy())

    # 【关键修改】将 Python 列表转换为 2D NumPy 数组
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # 此时 embeddings 拥有 .shape 属性（形状为 [总样本数, 128]）
    embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    for i in range(10):
        indices = [j for j, lbl in enumerate(labels) if lbl == i]
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=f'Class {i}', alpha=0.6, s=10)
    plt.legend()
    plt.title("t-SNE Visualization of Encoder Features")

    output_path = "tsne_embeddings.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE 降维可视化图片已成功保存至: {output_path}")
    plt.close()  # 关闭当前画布，释放内存

visualize_embeddings(encoder, val_loader)
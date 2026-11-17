import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 超参数设置
latent_dim = 100  # 生成器输入的噪声维度
image_dim = 28 * 28  # MNIST图像展开后的维度 (784)
lr = 0.0002
batch_size = 64
epochs = 50

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和判别器并移动到设备
generator = Generator(latent_dim, image_dim).to(device)
discriminator = Discriminator(image_dim).to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练循环
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        current_batch_size = real_images.size(0)
        real_images = real_images.view(current_batch_size, -1).to(device)

        # 创建真实标签和虚假标签
        real_labels = torch.ones(current_batch_size, 1).to(device)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        # 1. 训练判别器
        optimizer_D.zero_grad()
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)

        # 生成假样本
        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        # 总判别器损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 2. 训练生成器
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')


# 展示生成的图像
import os


def show_generated_images(generator, latent_dim, num_images=25, save_path="generated_images.png"):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = generator(z).view(-1, 1, 28, 28).cpu()

    grid_img = vutils.make_grid(generated_images, nrow=5, normalize=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid_img.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis('off')

    # 保存图像到本地
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭画布，释放内存
    print(f"图像已成功保存至: {os.path.abspath(save_path)}")


show_generated_images(generator, latent_dim)

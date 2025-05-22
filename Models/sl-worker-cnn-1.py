import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import numpy as np

#数据加载
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\data"
        self.transform = transform
        self.npy_paths = []  # 存储.npy文件路径
        self.labels = []  # 存储对应标签

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path) and label.isdigit():
                # 获取当前标签下所有.npy文件
                npy_files = glob.glob(os.path.join(label_path, "*.npy"))
                self.npy_paths.extend(npy_files)
                self.labels.extend([int(label)] * len(npy_files))

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        npy_path = self.npy_paths[idx]
        depth_images = np.load(npy_path).astype(np.float32)

        if depth_images.shape[0] != 3:
            raise ValueError(f"Invalid npy file {npy_path}: expected 3 time steps, got {depth_images.shape[0]}")

        if depth_images.dtype == np.uint8:
            depth_images = depth_images / 255.0  # 归一化到 [0,1]

        # 转换为 (3, H, W) 的 tensor
        image_tensor = torch.from_numpy(depth_images)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        label = self.labels[idx]  # 获取当前样本的标签
        return image_tensor, label

#注意力机制
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        mid_channels = int(in_channels * se_ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.Hardswish(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.squeeze(x).view(b, c, 1, 1)
        se = self.excitation(se)
        return x * se.expand_as(x)


class TimeFocusedModel(nn.Module):
    def __init__(self, num_classes=5, time_importance=[0.5, 0.3, 3.0], dropout_p=0.3):
        super().__init__()
        self.time_importance = nn.Parameter(
            torch.tensor(time_importance, dtype=torch.float32),
            requires_grad=True
        )

        # 空间卷积
        self.spatial_encoder = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardswish(),

            InvertedResidual(32, 32, stride=2, expand_ratio=4),

            InvertedResidual(32, 32, stride=1, expand_ratio=4),
            InvertedResidual(32, 64, stride=2, expand_ratio=4),

            InvertedResidual(64, 64, stride=1, expand_ratio=4),
            InvertedResidual(64, 128, stride=2, expand_ratio=4),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Conv2d(128, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1)
        )

        # 时间卷积
        self.time_raw_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),

            InvertedResidual(16, 24, stride=2, expand_ratio=4),

            InvertedResidual(24, 32, stride=2, expand_ratio=4),

            InvertedResidual(32, 64, stride=2, expand_ratio=4),

            nn.AdaptiveAvgPool2d(1)
        )

        # 时间序列1D卷积
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.3),

            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # 空间特征提取
        spatial_feat = self.spatial_encoder(x[:, 2:3, :, :])
        spatial_feat = spatial_feat.flatten(1)
        spatial_feat = spatial_feat * self.time_importance[2]

        # 时间特征提取
        time_raw_feats = []
        for t in range(3):
            raw_feat = self.time_raw_encoder(x[:, t:t + 1, :, :])
            raw_feat = raw_feat.flatten(1)
            raw_feat = raw_feat * self.time_importance[t]
            time_raw_feats.append(raw_feat)

        time_sequence = torch.stack(time_raw_feats, dim=2)
        time_feat = self.time_conv(time_sequence).squeeze(2)

        # 特征融合与分类
        combined_feat = torch.cat([spatial_feat, time_feat], dim=1)
        return self.classifier(combined_feat)


# 逆残差块
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.25):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expand_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, expand_channels, kernel_size=1),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(),

            nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, padding=1, groups=expand_channels),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(),

            SEBlock(expand_channels, se_ratio=se_ratio),

            nn.Conv2d(expand_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


#训练
def train():
    data_root = r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\data"  # 数据目录结构：train_data/标签/*.npy
    batch_size = 16
    num_epochs = 80
    lr = 0.0005
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Training & Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    # 归一化
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

    # 加载数据集
    dataset = ImageDataset(root_dir=data_root, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    unique_labels = dataset.labels.unique()
    num_classes = unique_labels.size(0)
    
    # 初始化模型
    model = TimeFocusedModel(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    epoch_losses = []
    epoch_val_losses = []
    epoch_accs = []
    epoch_val_accs = []

    best_val_loss = float('inf')
    best_val_acc = -float('inf')
    patience = 6
    no_improve_epochs = 0
    no_improve_acc_epochs = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 验证输入形状
            if batch_idx == 0 and epoch == 0:
                print(f"输入形状: {images.shape}（[batch, time_steps, H, W]）") # 应输出[32, 3, 128, 128]

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_accs.append(epoch_acc)
        epoch_losses.append(epoch_loss)

        print(f'\nEpoch [{epoch + 1}/{num_epochs}] Complete: '
              f'Avg Loss: {epoch_loss:.4f}, Avg Acc: {epoch_acc:.2f}%\n')

        # 验证损失
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss_total / len(val_loader)
        epoch_val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        val_acc = 100 * val_correct / len(val_dataset)
        epoch_val_accs.append(val_acc)
        print(f'Validation Acc: {val_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}')

        # 早停机制
        # if epoch_val_loss < best_val_loss:
        #     best_val_loss = epoch_val_loss
        #     no_improve_epochs = 0
        # else:
        #     no_improve_epochs += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()  
            no_improve_acc_epochs = 0
        else:
            no_improve_acc_epochs += 1

        if no_improve_epochs >= patience or no_improve_acc_epochs >= patience:
            print(f'\nEarly stopping triggered at epoch {epoch + 1}!')
            model.load_state_dict(best_model_weights)
            torch.save(best_model_weights, r"C:\Users\xiang\OneDrive\桌面\subwayai\pythonProject\subwAI-surfer\weights\CurrentBestModel.pth")
            break

        print(f'Best validation accuracy: {best_val_acc:.2f}')

        # 绘制损失曲线
        x = list(range(1, epoch + 2))
        ax.clear()
        ax.plot(x, epoch_losses, 'b-', label='Train Loss')
        ax.plot(x, epoch_val_losses, 'r-', label='Val Loss')
        ax.set_title('Training & Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)


    plt.ioff()
    plt.show()

    print('训练完成！')

if __name__ == '__main__':
    train()
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import pandas as pd


# MTCNN网络架构定义
class PNet(nn.Module):
    """Proposal Network (P-Net) - 12x12输入"""

    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU()

        # 分类分支
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        # 边界框回归分支
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # 关键点回归分支
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))

        cls = torch.sigmoid(self.conv4_1(x))
        bbox = self.conv4_2(x)
        landmark = self.conv4_3(x)

        return cls, bbox, landmark


class RNet(nn.Module):
    """Refine Network (R-Net) - 24x24输入"""

    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1)
        self.prelu3 = nn.PReLU()

        self.fc4 = nn.Linear(64 * 3 * 3, 128)
        self.prelu4 = nn.PReLU()

        # 分类分支
        self.fc5_1 = nn.Linear(128, 2)
        # 边界框回归分支
        self.fc5_2 = nn.Linear(128, 4)
        # 关键点回归分支
        self.fc5_3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.prelu3(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = self.prelu4(self.fc4(x))

        cls = torch.sigmoid(self.fc5_1(x))
        bbox = self.fc5_2(x)
        landmark = self.fc5_3(x)

        return cls, bbox, landmark


class ONet(nn.Module):
    """Output Network (O-Net) - 48x48输入"""

    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.prelu4 = nn.PReLU()

        self.fc5 = nn.Linear(128 * 3 * 3, 256)
        self.prelu5 = nn.PReLU()

        # 分类分支
        self.fc6_1 = nn.Linear(256, 2)
        # 边界框回归分支
        self.fc6_2 = nn.Linear(256, 4)
        # 关键点回归分支
        self.fc6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.pool3(self.prelu3(self.conv3(x)))
        x = self.prelu4(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.prelu5(self.fc5(x))

        cls = torch.sigmoid(self.fc6_1(x))
        bbox = self.fc6_2(x)
        landmark = self.fc6_3(x)

        return cls, bbox, landmark


# 数据预处理和增强
class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_file, bbox_file, landmark_file, net_type='PNet', transform=None):
        self.root_dir = root_dir
        self.net_type = net_type
        self.transform = transform

        # 读取CelebA标注文件
        self.attr_df = pd.read_csv(attr_file, delim_whitespace=True, header=1)
        self.bbox_df = pd.read_csv(bbox_file, delim_whitespace=True, header=1)
        self.landmark_df = pd.read_csv(landmark_file, delim_whitespace=True, header=1)

        # 根据网络类型设置输入尺寸
        if net_type == 'PNet':
            self.size = 12
        elif net_type == 'RNet':
            self.size = 24
        elif net_type == 'ONet':
            self.size = 48

        self.image_names = list(self.attr_df.index)

    def __len__(self):
        return len(self.image_names)

    def generate_sample(self, img, bbox, landmarks):
        """生成正样本、负样本和部分样本"""
        height, width = img.shape[:2]
        x, y, w, h = bbox

        samples = []
        labels = []
        bbox_targets = []
        landmark_targets = []

        # 生成正样本 (IoU > 0.65)
        for _ in range(10):
            # 随机偏移
            dx = np.random.uniform(-0.2, 0.2) * w
            dy = np.random.uniform(-0.2, 0.2) * h
            dw = np.random.uniform(-0.2, 0.2) * w
            dh = np.random.uniform(-0.2, 0.2) * h

            nx = max(0, x + dx)
            ny = max(0, y + dy)
            nw = min(width - nx, w + dw)
            nh = min(height - ny, h + dh)

            if nw > 0 and nh > 0:
                # 计算IoU
                iou = self.calculate_iou([x, y, w, h], [nx, ny, nw, nh])
                if iou > 0.65:
                    crop = img[int(ny):int(ny + nh), int(nx):int(nx + nw)]
                    crop = cv2.resize(crop, (self.size, self.size))

                    # 计算相对偏移
                    offset_x = (x - nx) / nw
                    offset_y = (y - ny) / nh
                    offset_w = w / nw
                    offset_h = h / nh

                    samples.append(crop)
                    labels.append(1)  # 正样本
                    bbox_targets.append([offset_x, offset_y, offset_w, offset_h])
                    landmark_targets.append(self.transform_landmarks(landmarks, nx, ny, nw, nh))

        # 生成负样本 (IoU < 0.3)
        for _ in range(20):
            size = np.random.uniform(0.3, 1.0) * min(width, height)
            nx = np.random.uniform(0, width - size)
            ny = np.random.uniform(0, height - size)

            iou = self.calculate_iou([x, y, w, h], [nx, ny, size, size])
            if iou < 0.3:
                crop = img[int(ny):int(ny + size), int(nx):int(nx + size)]
                crop = cv2.resize(crop, (self.size, self.size))

                samples.append(crop)
                labels.append(0)  # 负样本
                bbox_targets.append([0, 0, 0, 0])
                landmark_targets.append([0] * 10)

        # 生成部分样本 (0.4 < IoU < 0.65)
        for _ in range(5):
            dx = np.random.uniform(-0.5, 0.5) * w
            dy = np.random.uniform(-0.5, 0.5) * h
            dw = np.random.uniform(-0.3, 0.3) * w
            dh = np.random.uniform(-0.3, 0.3) * h

            nx = max(0, x + dx)
            ny = max(0, y + dy)
            nw = min(width - nx, w + dw)
            nh = min(height - ny, h + dh)

            if nw > 0 and nh > 0:
                iou = self.calculate_iou([x, y, w, h], [nx, ny, nw, nh])
                if 0.4 < iou < 0.65:
                    crop = img[int(ny):int(ny + nh), int(nx):int(nx + nw)]
                    crop = cv2.resize(crop, (self.size, self.size))

                    offset_x = (x - nx) / nw
                    offset_y = (y - ny) / nh
                    offset_w = w / nw
                    offset_h = h / nh

                    samples.append(crop)
                    labels.append(-1)  # 部分样本
                    bbox_targets.append([offset_x, offset_y, offset_w, offset_h])
                    landmark_targets.append(self.transform_landmarks(landmarks, nx, ny, nw, nh))

        return samples, labels, bbox_targets, landmark_targets

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

    def transform_landmarks(self, landmarks, x, y, w, h):
        """将关键点坐标转换为相对坐标"""
        transformed = []
        for i in range(0, len(landmarks), 2):
            lx = (landmarks[i] - x) / w
            ly = (landmarks[i + 1] - y) / h
            transformed.extend([lx, ly])
        return transformed

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, 'img_align_celeba', img_name)

        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 获取边界框和关键点
        bbox_row = self.bbox_df.loc[img_name]
        bbox = [bbox_row['x_1'], bbox_row['y_1'], bbox_row['width'], bbox_row['height']]

        landmark_row = self.landmark_df.loc[img_name]
        landmarks = [landmark_row[f'lefteye_x'], landmark_row[f'lefteye_y'],
                     landmark_row[f'righteye_x'], landmark_row[f'righteye_y'],
                     landmark_row[f'nose_x'], landmark_row[f'nose_y'],
                     landmark_row[f'leftmouth_x'], landmark_row[f'leftmouth_y'],
                     landmark_row[f'rightmouth_x'], landmark_row[f'rightmouth_y']]

        # 生成训练样本
        samples, labels, bbox_targets, landmark_targets = self.generate_sample(img, bbox, landmarks)

        # 随机选择一个样本
        if samples:
            idx = np.random.randint(len(samples))
            sample = samples[idx]
            label = labels[idx]
            bbox_target = bbox_targets[idx]
            landmark_target = landmark_targets[idx]

            if self.transform:
                sample = self.transform(sample)
            else:
                sample = torch.FloatTensor(sample).permute(2, 0, 1) / 255.0

            return {
                'image': sample,
                'label': torch.FloatTensor([label]),
                'bbox': torch.FloatTensor(bbox_target),
                'landmark': torch.FloatTensor(landmark_target)
            }
        else:
            # 如果没有生成有效样本，返回原始裁剪
            x, y, w, h = bbox
            crop = img[int(y):int(y + h), int(x):int(x + w)]
            crop = cv2.resize(crop, (self.size, self.size))

            if self.transform:
                crop = self.transform(crop)
            else:
                crop = torch.FloatTensor(crop).permute(2, 0, 1) / 255.0

            return {
                'image': crop,
                'label': torch.FloatTensor([1]),
                'bbox': torch.FloatTensor([0, 0, 1, 1]),
                'landmark': torch.FloatTensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
            }


# 损失函数
class MTCNNLoss(nn.Module):
    def __init__(self):
        super(MTCNNLoss, self).__init__()
        self.cls_loss = nn.BCELoss()
        self.bbox_loss = nn.MSELoss()
        self.landmark_loss = nn.MSELoss()

    def forward(self, cls_pred, bbox_pred, landmark_pred, cls_target, bbox_target, landmark_target):
        # 分类损失
        cls_mask = (cls_target >= 0).float()  # 忽略部分样本的分类损失
        cls_loss = self.cls_loss(cls_pred.squeeze() * cls_mask, cls_target.squeeze() * cls_mask)

        # 边界框回归损失 (只对正样本和部分样本计算)
        bbox_mask = (cls_target != 0).float().unsqueeze(1)
        bbox_loss = self.bbox_loss(bbox_pred * bbox_mask, bbox_target * bbox_mask)

        # 关键点回归损失 (只对正样本计算)
        landmark_mask = (cls_target == 1).float().unsqueeze(1)
        landmark_loss = self.landmark_loss(landmark_pred * landmark_mask, landmark_target * landmark_mask)

        total_loss = cls_loss + 0.5 * bbox_loss + 0.5 * landmark_loss

        return total_loss, cls_loss, bbox_loss, landmark_loss


# 训练函数
class MTCNNTrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 数据变换
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
        ])

    def train_network(self, net_type='PNet', epochs=50, batch_size=32, lr=0.001):
        """训练指定的网络"""
        print(f"Training {net_type}...")

        # 创建网络
        if net_type == 'PNet':
            model = PNet()
        elif net_type == 'RNet':
            model = RNet()
        elif net_type == 'ONet':
            model = ONet()

        model = model.to(self.device)

        # 创建数据集和数据加载器
        dataset = CelebADataset(
            root_dir=self.data_dir,
            attr_file=os.path.join(self.data_dir, 'list_attr_celeba.txt'),
            bbox_file=os.path.join(self.data_dir, 'list_bbox_celeba.txt'),
            landmark_file=os.path.join(self.data_dir, 'list_landmarks_align_celeba.txt'),
            net_type=net_type,
            transform=self.transform
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = MTCNNLoss()

        # 训练循环
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            cls_loss_sum = 0
            bbox_loss_sum = 0
            landmark_loss_sum = 0

            pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                cls_labels = batch['label'].to(self.device)
                bbox_labels = batch['bbox'].to(self.device)
                landmark_labels = batch['landmark'].to(self.device)

                optimizer.zero_grad()

                # 前向传播
                cls_pred, bbox_pred, landmark_pred = model(images)

                # 计算损失
                loss, cls_loss, bbox_loss, landmark_loss = criterion(
                    cls_pred, bbox_pred, landmark_pred,
                    cls_labels, bbox_labels, landmark_labels
                )

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cls_loss_sum += cls_loss.item()
                bbox_loss_sum += bbox_loss.item()
                landmark_loss_sum += landmark_loss.item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Cls': f'{cls_loss.item():.4f}',
                    'Box': f'{bbox_loss.item():.4f}',
                    'Landmark': f'{landmark_loss.item():.4f}'
                })

            avg_loss = total_loss / len(dataloader)
            avg_cls_loss = cls_loss_sum / len(dataloader)
            avg_bbox_loss = bbox_loss_sum / len(dataloader)
            avg_landmark_loss = landmark_loss_sum / len(dataloader)

            print(f'Epoch {epoch + 1}: Loss={avg_loss:.4f}, Cls={avg_cls_loss:.4f}, '
                  f'Box={avg_bbox_loss:.4f}, Landmark={avg_landmark_loss:.4f}')

            # 保存模型
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'{net_type}_epoch_{epoch + 1}.pth')

        # 保存最终模型
        torch.save(model.state_dict(), f'{net_type}_final.pth')
        print(f"{net_type} training completed!")

        return model

    def train_cascade(self):
        """级联训练所有网络"""
        print("Starting MTCNN cascade training...")

        # 训练P-Net
        pnet = self.train_network('PNet', epochs=30, batch_size=64, lr=0.001)

        # 训练R-Net
        rnet = self.train_network('RNet', epochs=40, batch_size=32, lr=0.0005)

        # 训练O-Net
        onet = self.train_network('ONet', epochs=50, batch_size=16, lr=0.0001)

        print("MTCNN cascade training completed!")
        return pnet, rnet, onet


# 使用示例
if __name__ == "__main__":
    # CelebA数据集路径
    data_dir = "/path/to/celeba"  # 请修改为您的CelebA数据集路径

    # 创建训练器
    trainer = MTCNNTrainer(data_dir)

    # 开始训练
    # 可以选择训练单个网络或整个级联

    # 训练单个网络
    # pnet = trainer.train_network('PNet', epochs=30)

    # 或者训练整个级联
    pnet, rnet, onet = trainer.train_cascade()
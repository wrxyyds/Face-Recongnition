import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入MTCNN模型
from face.mtcnn.model import PNet, RNet, ONet
from face.mtcnn.box_utils import convert_to_square, calibrate_box, get_image_boxes


class CelebADataset(Dataset):

    def __init__(self, root_dir, anno_file, transform=None, max_samples=1000):
        self.root_dir = root_dir
        self.transform = transform
        self.max_samples = max_samples

        # 加载注释文件
        self.annotations = self.load_annotations(anno_file)

        # 限制样本数量
        if len(self.annotations) > max_samples:
            self.annotations = self.annotations[:max_samples]

    def load_annotations(self, anno_file):
        """加载CelebA人脸边界框注释"""
        annotations = []

        # 如果注释文件存在，直接加载
        if os.path.exists(anno_file):
            with open(anno_file, 'r') as f:
                annotations = json.load(f)
        else:
            # 生成伪注释（实际使用时需要真实的标注数据）
            img_dir = os.path.join(self.root_dir, 'img_align_celeba')
            if os.path.exists(img_dir):
                img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')][:self.max_samples]

                for img_file in img_files:
                    # 为演示目的生成伪边界框（实际训练需要真实标注）
                    img_path = os.path.join(img_dir, img_file)
                    try:
                        img = Image.open(img_path)
                        w, h = img.size

                        # 生成中心区域的伪人脸框
                        x1 = w * 0.2
                        y1 = h * 0.2
                        x2 = w * 0.8
                        y2 = h * 0.8

                        annotations.append({
                            'filename': img_file,
                            'bbox': [x1, y1, x2, y2],
                            'landmarks': [
                                w * 0.35, w * 0.65, w * 0.5, w * 0.35, w * 0.65,  # x坐标
                                h * 0.4, h * 0.4, h * 0.55, h * 0.7, h * 0.7  # y坐标
                            ]
                        })
                    except Exception as e:
                        print(f"处理图片 {img_file} 时出错: {e}")
                        continue

                # 保存生成的注释
                os.makedirs(os.path.dirname(anno_file), exist_ok=True)
                with open(anno_file, 'w') as f:
                    json.dump(annotations, f, indent=2)

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.root_dir, 'img_align_celeba', annotation['filename'])

        # 加载图片
        image = Image.open(img_path).convert('RGB')
        bbox = annotation['bbox']
        landmarks = annotation['landmarks']

        # 应用数据增强
        if self.transform:
            image, bbox, landmarks = self.transform(image, bbox, landmarks)

        return image, bbox, landmarks, annotation['filename']


class DataAugmentation:
    """数据增强类"""

    def __init__(self, rotation_range=10, flip_prob=0.5, color_enhance_prob=0.3):
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.color_enhance_prob = color_enhance_prob

    def __call__(self, image, bbox, landmarks):
        # 随机小角度旋转
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle, expand=False)

        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w = image.size[0]
            # 调整边界框
            bbox[0], bbox[2] = w - bbox[2], w - bbox[0]
            # 调整关键点
            landmarks[0], landmarks[1], landmarks[3], landmarks[4] = \
                w - landmarks[1], w - landmarks[0], w - landmarks[4], w - landmarks[3]

        # 随机颜色增强
        if random.random() < self.color_enhance_prob:
            # 亮度调整
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

            # 对比度调整
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

            # 饱和度调整
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        return image, bbox, landmarks


class MTCNNTrainer:
    """MTCNN训练器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)

        # 损失函数
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.bbox_loss_fn = nn.MSELoss(reduction='none')
        self.landmark_loss_fn = nn.MSELoss(reduction='none')

        # 优化器
        self.pnet_optimizer = optim.Adam(self.pnet.parameters(), lr=0.001)
        self.rnet_optimizer = optim.Adam(self.rnet.parameters(), lr=0.001)
        self.onet_optimizer = optim.Adam(self.onet.parameters(), lr=0.001)

        # 学习率调度器
        self.pnet_scheduler = optim.lr_scheduler.StepLR(self.pnet_optimizer, step_size=10, gamma=0.5)
        self.rnet_scheduler = optim.lr_scheduler.StepLR(self.rnet_optimizer, step_size=10, gamma=0.5)
        self.onet_scheduler = optim.lr_scheduler.StepLR(self.onet_optimizer, step_size=10, gamma=0.5)

    def generate_training_data(self, dataloader, stage='pnet'):
        """为不同网络阶段生成训练数据"""
        positive_samples = []
        negative_samples = []
        part_samples = []

        for batch_idx, (images, bboxes, landmarks, filenames) in enumerate(
                tqdm(dataloader, desc=f"生成{stage}训练数据")):
            for i in range(len(images)):
                image = images[i]
                bbox = bboxes[i]
                landmark = landmarks[i]

                # 生成正样本、负样本和部分样本
                pos_samples, neg_samples, part_samples_batch = self._generate_samples_for_image(
                    image, bbox, landmark, stage)

                positive_samples.extend(pos_samples)
                negative_samples.extend(neg_samples)
                part_samples.extend(part_samples_batch)

        return positive_samples, negative_samples, part_samples

    def _generate_samples_for_image(self, image, bbox, landmark, stage):
        """为单张图片生成训练样本"""
        pos_samples = []
        neg_samples = []
        part_samples = []

        w, h = image.size

        # 根据不同阶段设置样本大小
        if stage == 'pnet':
            sample_size = 12
        elif stage == 'rnet':
            sample_size = 24
        else:  # onet
            sample_size = 48

        # 生成正样本 (IoU > 0.65)
        for _ in range(20):  # 每张图片生成20个正样本
            # 在真实边界框附近采样
            noise_x = random.uniform(-0.1, 0.1) * (bbox[2] - bbox[0])
            noise_y = random.uniform(-0.1, 0.1) * (bbox[3] - bbox[1])
            noise_w = random.uniform(-0.1, 0.1) * (bbox[2] - bbox[0])
            noise_h = random.uniform(-0.1, 0.1) * (bbox[3] - bbox[1])

            sample_x1 = max(0, bbox[0] + noise_x)
            sample_y1 = max(0, bbox[1] + noise_y)
            sample_x2 = min(w, bbox[2] + noise_w)
            sample_y2 = min(h, bbox[3] + noise_h)

            if self._calculate_iou([sample_x1, sample_y1, sample_x2, sample_y2], bbox) > 0.65:
                crop_img = image.crop((sample_x1, sample_y1, sample_x2, sample_y2))
                crop_img = crop_img.resize((sample_size, sample_size), Image.BILINEAR)
                pos_samples.append((crop_img, 1, [sample_x1, sample_y1, sample_x2, sample_y2], landmark))

        # 生成负样本 (IoU < 0.3)
        for _ in range(50):  # 每张图片生成50个负样本
            sample_x1 = random.uniform(0, w - sample_size)
            sample_y1 = random.uniform(0, h - sample_size)
            sample_x2 = sample_x1 + random.uniform(sample_size, min(w - sample_x1, sample_size * 2))
            sample_y2 = sample_y1 + random.uniform(sample_size, min(h - sample_y1, sample_size * 2))

            if self._calculate_iou([sample_x1, sample_y1, sample_x2, sample_y2], bbox) < 0.3:
                crop_img = image.crop((sample_x1, sample_y1, sample_x2, sample_y2))
                crop_img = crop_img.resize((sample_size, sample_size), Image.BILINEAR)
                neg_samples.append((crop_img, 0, [sample_x1, sample_y1, sample_x2, sample_y2], landmark))

        # 生成部分样本 (0.4 < IoU < 0.65)
        for _ in range(20):  # 每张图片生成20个部分样本
            noise_x = random.uniform(-0.3, 0.3) * (bbox[2] - bbox[0])
            noise_y = random.uniform(-0.3, 0.3) * (bbox[3] - bbox[1])

            sample_x1 = max(0, bbox[0] + noise_x)
            sample_y1 = max(0, bbox[1] + noise_y)
            sample_x2 = min(w, bbox[2] + noise_x)
            sample_y2 = min(h, bbox[3] + noise_y)

            iou = self._calculate_iou([sample_x1, sample_y1, sample_x2, sample_y2], bbox)
            if 0.4 < iou < 0.65:
                crop_img = image.crop((sample_x1, sample_y1, sample_x2, sample_y2))
                crop_img = crop_img.resize((sample_size, sample_size), Image.BILINEAR)
                part_samples.append((crop_img, -1, [sample_x1, sample_y1, sample_x2, sample_y2], landmark))

        return pos_samples, neg_samples, part_samples

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def online_hard_example_mining(self, losses, ratio=0.7):
        """在线难例挖掘：选择前70%的高损失样本"""
        num_samples = len(losses)
        num_hard = int(num_samples * ratio)

        # 获取损失值最高的样本索引
        sorted_indices = torch.argsort(losses, descending=True)
        hard_indices = sorted_indices[:num_hard]

        return hard_indices

    def train_pnet(self, train_dataloader, val_dataloader, epochs=20):
        """训练P-Net"""
        print("开始训练P-Net...")
        best_val_loss = float('inf')
        best_pnet_state = None

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.pnet.train()
            total_loss = 0.0
            num_batches = 0

            # 生成训练数据
            pos_samples, neg_samples, part_samples = self.generate_training_data(train_dataloader, 'pnet')
            all_samples = pos_samples + neg_samples + part_samples
            random.shuffle(all_samples)

            # 批处理训练
            batch_size = 128
            for i in tqdm(range(0, len(all_samples), batch_size), desc=f"P-Net Epoch {epoch + 1}"):
                batch_samples = all_samples[i:i + batch_size]

                # 准备批数据
                images = []
                labels = []
                bboxes = []

                for sample in batch_samples:
                    img, label, bbox, _ = sample
                    img_array = np.array(img).astype(np.float32)
                    img_array = (img_array - 127.5) * 0.0078125
                    img_array = np.transpose(img_array, (2, 0, 1))
                    images.append(img_array)
                    labels.append(1 if label == 1 else 0)  # 二分类：人脸/非人脸
                    bboxes.append(bbox)

                images = torch.FloatTensor(images).to(self.device)
                labels = torch.LongTensor(labels).to(self.device)

                # 前向传播
                bbox_pred, cls_pred = self.pnet(images)

                # 计算损失
                cls_loss = self.cls_loss_fn(cls_pred.view(-1, 2), labels)

                # 在线难例挖掘
                hard_indices = self.online_hard_example_mining(cls_loss)
                cls_loss = cls_loss[hard_indices].mean()

                # 反向传播
                self.pnet_optimizer.zero_grad()
                cls_loss.backward()
                self.pnet_optimizer.step()

                total_loss += cls_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"P-Net Epoch {epoch + 1}/{epochs}, Training Average Loss: {avg_loss:.4f}")
            train_losses.append(avg_loss)

            # 验证集评估
            self.pnet.eval()
            val_total_loss = 0.0
            val_num_batches = 0
            with torch.no_grad():
                pos_samples, neg_samples, part_samples = self.generate_training_data(val_dataloader, 'pnet')
                all_samples = pos_samples + neg_samples + part_samples
                random.shuffle(all_samples)
                for i in range(0, len(all_samples), batch_size):
                    batch_samples = all_samples[i:i + batch_size]
                    images = []
                    labels = []
                    for sample in batch_samples:
                        img, label, _, _ = sample
                        img_array = np.array(img).astype(np.float32)
                        img_array = (img_array - 127.5) * 0.0078125
                        img_array = np.transpose(img_array, (2, 0, 1))
                        images.append(img_array)
                        labels.append(1 if label == 1 else 0)
                    images = torch.FloatTensor(images).to(self.device)
                    labels = torch.LongTensor(labels).to(self.device)
                    bbox_pred, cls_pred = self.pnet(images)
                    cls_loss = self.cls_loss_fn(cls_pred.view(-1, 2), labels)
                    hard_indices = self.online_hard_example_mining(cls_loss)
                    cls_loss = cls_loss[hard_indices].mean()
                    val_total_loss += cls_loss.item()
                    val_num_batches += 1
            avg_val_loss = val_total_loss / val_num_batches if val_num_batches > 0 else 0
            print(f"P-Net Epoch {epoch + 1}/{epochs}, Validation Average Loss: {avg_val_loss:.4f}")
            val_losses.append(avg_val_loss)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_pnet_state = self.pnet.state_dict()

            self.pnet_scheduler.step()

        # 加载最佳模型
        self.pnet.load_state_dict(best_pnet_state)
        # 保存模型
        torch.save(self.pnet.state_dict(), 'pnet_trained.pth')
        print("P-Net训练完成并保存")

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.show()

    def train_rnet(self, train_dataloader, val_dataloader, epochs=20):
        """训练R-Net"""
        print("开始训练R-Net...")
        best_val_loss = float('inf')
        best_rnet_state = None

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.rnet.train()
            total_loss = 0.0
            num_batches = 0

            # 生成训练数据
            pos_samples, neg_samples, part_samples = self.generate_training_data(train_dataloader, 'rnet')
            all_samples = pos_samples + neg_samples + part_samples
            random.shuffle(all_samples)

            # 批处理训练
            batch_size = 128
            for i in tqdm(range(0, len(all_samples), batch_size), desc=f"R-Net Epoch {epoch + 1}"):
                batch_samples = all_samples[i:i + batch_size]

                # 准备批数据
                images = []
                labels = []

                for sample in batch_samples:
                    img, label, _, _ = sample
                    img_array = np.array(img).astype(np.float32)
                    img_array = (img_array - 127.5) * 0.0078125
                    img_array = np.transpose(img_array, (2, 0, 1))
                    images.append(img_array)
                    labels.append(1 if label == 1 else 0)

                images = torch.FloatTensor(images).to(self.device)
                labels = torch.LongTensor(labels).to(self.device)

                # 前向传播
                bbox_pred, cls_pred = self.rnet(images)

                # 计算损失
                cls_loss = self.cls_loss_fn(cls_pred, labels)

                # 在线难例挖掘
                hard_indices = self.online_hard_example_mining(cls_loss)
                cls_loss = cls_loss[hard_indices].mean()

                # 反向传播
                self.rnet_optimizer.zero_grad()
                cls_loss.backward()
                self.rnet_optimizer.step()

                total_loss += cls_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"R-Net Epoch {epoch + 1}/{epochs}, Training Average Loss: {avg_loss:.4f}")
            train_losses.append(avg_loss)

            # 验证集评估
            self.rnet.eval()
            val_total_loss = 0.0
            val_num_batches = 0
            with torch.no_grad():
                pos_samples, neg_samples, part_samples = self.generate_training_data(val_dataloader, 'rnet')
                all_samples = pos_samples + neg_samples + part_samples
                random.shuffle(all_samples)
                for i in range(0, len(all_samples), batch_size):
                    batch_samples = all_samples[i:i + batch_size]
                    images = []
                    labels = []
                    for sample in batch_samples:
                        img, label, _, _ = sample
                        img_array = np.array(img).astype(np.float32)
                        img_array = (img_array - 127.5) * 0.0078125
                        img_array = np.transpose(img_array, (2, 0, 1))
                        images.append(img_array)
                        labels.append(1 if label == 1 else 0)
                    images = torch.FloatTensor(images).to(self.device)
                    labels = torch.LongTensor(labels).to(self.device)
                    bbox_pred, cls_pred = self.rnet(images)
                    cls_loss = self.cls_loss_fn(cls_pred, labels)
                    hard_indices = self.online_hard_example_mining(cls_loss)
                    cls_loss = cls_loss[hard_indices].mean()
                    val_total_loss += cls_loss.item()
                    val_num_batches += 1
            avg_val_loss = val_total_loss / val_num_batches if val_num_batches > 0 else 0
            print(f"R-Net Epoch {epoch + 1}/{epochs}, Validation Average Loss: {avg_val_loss:.4f}")
            val_losses.append(avg_val_loss)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_rnet_state = self.rnet.state_dict()

            self.rnet_scheduler.step()

        # 加载最佳模型
        self.rnet.load_state_dict(best_rnet_state)
        # 保存模型
        torch.save(self.rnet.state_dict(), 'rnet_trained.pth')
        print("R-Net训练完成并保存")

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Loss')
        plt.legend()
        plt.show()

    def train_onet(self, train_dataloader, val_dataloader, epochs=20):
        """训练O-Net"""
        print("开始训练O-Net...")
        best_val_loss = float('inf')
        best_onet_state = None

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.onet.train()
            total_cls_loss = 0.0
            total_bbox_loss = 0.0
            total_landmark_loss = 0.0
            num_batches = 0

            # 生成训练数据
            pos_samples, neg_samples, part_samples = self.generate_training_data(train_dataloader, 'onet')
            all_samples = pos_samples + neg_samples + part_samples
            random.shuffle(all_samples)

            # 批处理训练
            batch_size = 64
            for i in tqdm(range(0, len(all_samples), batch_size), desc=f"O-Net Epoch {epoch + 1}"):
                batch_samples = all_samples[i:i + batch_size]

                # 准备批数据
                images = []
                labels = []
                bboxes = []
                landmarks = []

                for sample in batch_samples:
                    img, label, bbox, landmark = sample
                    img_array = np.array(img).astype(np.float32)
                    img_array = (img_array - 127.5) * 0.0078125
                    img_array = np.transpose(img_array, (2, 0, 1))
                    images.append(img_array)
                    labels.append(1 if label == 1 else 0)
                    bboxes.append(bbox)
                    landmarks.append(landmark)

                images = torch.FloatTensor(images).to(self.device)
                labels = torch.LongTensor(labels).to(self.device)
                bboxes = torch.FloatTensor(bboxes).to(self.device)
                landmarks = torch.FloatTensor(landmarks).to(self.device)

                # 前向传播
                landmark_pred, bbox_pred, cls_pred = self.onet(images)

                # 计算损失
                cls_loss = self.cls_loss_fn(cls_pred, labels)
                bbox_loss = self.bbox_loss_fn(bbox_pred, bboxes).mean(dim=1)
                landmark_loss = self.landmark_loss_fn(landmark_pred, landmarks).mean(dim=1)

                # 总损失
                total_loss = cls_loss + bbox_loss + landmark_loss

                # 在线难例挖掘
                hard_indices = self.online_hard_example_mining(total_loss)
                cls_loss = cls_loss[hard_indices].mean()
                bbox_loss = bbox_loss[hard_indices].mean()
                landmark_loss = landmark_loss[hard_indices].mean()

                final_loss = cls_loss + bbox_loss + landmark_loss

                # 反向传播
                self.onet_optimizer.zero_grad()
                final_loss.backward()
                self.onet_optimizer.step()

                total_cls_loss += cls_loss.item()
                total_bbox_loss += bbox_loss.item()
                total_landmark_loss += landmark_loss.item()
                num_batches += 1

            avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0
            avg_bbox_loss = total_bbox_loss / num_batches if num_batches > 0 else 0
            avg_landmark_loss = total_landmark_loss / num_batches if num_batches > 0 else 0

            total_train_loss = avg_cls_loss + avg_bbox_loss + avg_landmark_loss
            print(f"O-Net Epoch {epoch + 1}/{epochs}")
            print(f"  Training Cls Loss: {avg_cls_loss:.4f}")
            print(f"  Training BBox Loss: {avg_bbox_loss:.4f}")
            print(f"  Training Landmark Loss: {avg_landmark_loss:.4f}")
            train_losses.append(total_train_loss)

            # 验证集评估
            self.onet.eval()
            val_total_cls_loss = 0.0
            val_total_bbox_loss = 0.0
            val_total_landmark_loss = 0.0
            val_num_batches = 0
            with torch.no_grad():
                pos_samples, neg_samples, part_samples = self.generate_training_data(val_dataloader, 'onet')
                all_samples = pos_samples + neg_samples + part_samples
                random.shuffle(all_samples)
                for i in range(0, len(all_samples), batch_size):
                    batch_samples = all_samples[i:i + batch_size]
                    images = []
                    labels = []
                    bboxes = []
                    landmarks = []
                    for sample in batch_samples:
                        img, label, bbox, landmark = sample
                        img_array = np.array(img).astype(np.float32)
                        img_array = (img_array - 127.5) * 0.0078125
                        img_array = np.transpose(img_array, (2, 0, 1))
                        images.append(img_array)
                        labels.append(1 if label == 1 else 0)
                        bboxes.append(bbox)
                        landmarks.append(landmark)
                    images = torch.FloatTensor(images).to(self.device)
                    labels = torch.LongTensor(labels).to(self.device)
                    bboxes = torch.FloatTensor(bboxes).to(self.device)
                    landmarks = torch.FloatTensor(landmarks).to(self.device)
                    landmark_pred, bbox_pred, cls_pred = self.onet(images)
                    cls_loss = self.cls_loss_fn(cls_pred, labels)
                    bbox_loss = self.bbox_loss_fn(bbox_pred, bboxes).mean(dim=1)
                    landmark_loss = self.landmark_loss_fn(landmark_pred, landmarks).mean(dim=1)
                    total_loss = cls_loss + bbox_loss + landmark_loss
                    hard_indices = self.online_hard_example_mining(total_loss)
                    cls_loss = cls_loss[hard_indices].mean()
                    bbox_loss = bbox_loss[hard_indices].mean()
                    landmark_loss = landmark_loss[hard_indices].mean()
                    val_total_cls_loss += cls_loss.item()
                    val_total_bbox_loss += bbox_loss.item()
                    val_total_landmark_loss += landmark_loss.item()
                    val_num_batches += 1
            avg_val_cls_loss = val_total_cls_loss / val_num_batches if val_num_batches > 0 else 0
            avg_val_bbox_loss = val_total_bbox_loss / val_num_batches if val_num_batches > 0 else 0
            avg_val_landmark_loss = val_total_landmark_loss / val_num_batches if val_num_batches > 0 else 0

            total_val_loss = avg_val_cls_loss + avg_val_bbox_loss + avg_val_landmark_loss
            print(f"O-Net Epoch {epoch + 1}/{epochs}")
            print(f"  Validation Cls Loss: {avg_val_cls_loss:.4f}")
            print(f"  Validation BBox Loss: {avg_val_bbox_loss:.4f}")
            print(f"  Validation Landmark Loss: {avg_val_landmark_loss:.4f}")
            val_losses.append(total_val_loss)

            # 保存最佳模型
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_onet_state = self.onet.state_dict()

            self.onet_scheduler.step()

        # 加载最佳模型
        self.onet.load_state_dict(best_onet_state)
        # 保存模型
        torch.save(self.onet.state_dict(), 'onet_trained.pth')
        print("O-Net训练完成并保存")

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('O-Net Model Loss')
        plt.legend()
        plt.show()

    def test_model(self, test_dataloader, stage='pnet'):
        """测试模型"""
        if stage == 'pnet':
            model = self.pnet
        elif stage == 'rnet':
            model = self.rnet
        else:
            model = self.onet
        model.eval()
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        total_landmark_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            pos_samples, neg_samples, part_samples = self.generate_training_data(test_dataloader, stage)
            all_samples = pos_samples + neg_samples + part_samples
            random.shuffle(all_samples)
            if stage == 'pnet' or stage == 'rnet':
                batch_size = 128
            else:
                batch_size = 64
            for i in range(0, len(all_samples), batch_size):
                batch_samples = all_samples[i:i + batch_size]
                images = []
                labels = []
                bboxes = []
                landmarks = []
                for sample in batch_samples:
                    img, label, bbox, landmark = sample
                    img_array = np.array(img).astype(np.float32)
                    img_array = (img_array - 127.5) * 0.0078125
                    img_array = np.transpose(img_array, (2, 0, 1))
                    images.append(img_array)
                    labels.append(1 if label == 1 else 0)
                    bboxes.append(bbox)
                    landmarks.append(landmark)
                images = torch.FloatTensor(images).to(self.device)
                labels = torch.LongTensor(labels).to(self.device)
                bboxes = torch.FloatTensor(bboxes).to(self.device)
                landmarks = torch.FloatTensor(landmarks).to(self.device)
                if stage == 'pnet':
                    bbox_pred, cls_pred = model(images)
                    cls_loss = self.cls_loss_fn(cls_pred.view(-1, 2), labels)
                    hard_indices = self.online_hard_example_mining(cls_loss)
                    cls_loss = cls_loss[hard_indices].mean()
                    total_cls_loss += cls_loss.item()
                elif stage == 'rnet':
                    bbox_pred, cls_pred = model(images)
                    cls_loss = self.cls_loss_fn(cls_pred, labels)
                    hard_indices = self.online_hard_example_mining(cls_loss)
                    cls_loss = cls_loss[hard_indices].mean()
                    total_cls_loss += cls_loss.item()
                else:
                    landmark_pred, bbox_pred, cls_pred = model(images)
                    cls_loss = self.cls_loss_fn(cls_pred, labels)
                    bbox_loss = self.bbox_loss_fn(bbox_pred, bboxes).mean(dim=1)
                    landmark_loss = self.landmark_loss_fn(landmark_pred, landmarks).mean(dim=1)
                    total_loss = cls_loss + bbox_loss + landmark_loss
                    hard_indices = self.online_hard_example_mining(total_loss)
                    cls_loss = cls_loss[hard_indices].mean()
                    bbox_loss = bbox_loss[hard_indices].mean()
                    landmark_loss = landmark_loss[hard_indices].mean()
                    total_cls_loss += cls_loss.item()
                    total_bbox_loss += bbox_loss.item()
                    total_landmark_loss += landmark_loss.item()
                num_batches += 1
        if stage == 'pnet' or stage == 'rnet':
            avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0
            print(f"{stage.upper()}-Net Test Cls Loss: {avg_cls_loss:.4f}")
        else:
            avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0
            avg_bbox_loss = total_bbox_loss / num_batches if num_batches > 0 else 0
            avg_landmark_loss = total_landmark_loss / num_batches if num_batches > 0 else 0
            print(f"O-Net Test Cls Loss: {avg_cls_loss:.4f}")
            print(f"O-Net Test BBox Loss: {avg_bbox_loss:.4f}")
            print(f"O-Net Test Landmark Loss: {avg_landmark_loss:.4f}")


def main():
    """主训练函数"""
    # 设置数据路径
    celeba_root = './celeba'  # CelebA数据集根目录
    anno_file = './celeba/annotations.json'  # 注释文件路径

    # 检查数据集是否存在
    if not os.path.exists(celeba_root):
        print(f"错误：CelebA数据集路径不存在: {celeba_root}")
        print("请下载CelebA数据集并将其放置在正确的路径下")
        return

    # 创建数据增强
    data_aug = DataAugmentation(rotation_range=10, flip_prob=0.5, color_enhance_prob=0.3)

    # 创建数据集
    dataset = CelebADataset(
        root_dir=celeba_root,
        anno_file=anno_file,
        transform=data_aug,
        max_samples=1000
    )

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建训练器
    trainer = MTCNNTrainer()

    # 逐步训练各个网络
    print("=" * 50)
    trainer.train_pnet(train_dataloader, val_dataloader, epochs=15)

    print("=" * 50)
    trainer.train_rnet(train_dataloader, val_dataloader, epochs=15)

    print("=" * 50)
    trainer.train_onet(train_dataloader, val_dataloader, epochs=15)

    # 测试模型
    print("=" * 50)
    print("开始测试模型...")
    trainer.test_model(test_dataloader, stage='pnet')
    trainer.test_model(test_dataloader, stage='rnet')
    trainer.test_model(test_dataloader, stage='onet')

    print("训练完成！模型已保存为：")
    print("- pnet_trained.pth")
    print("- rnet_trained.pth")
    print("- onet_trained.pth")


if __name__ == "__main__":
    main()
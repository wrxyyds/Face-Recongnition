import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PySide6.QtWidgets import QProgressDialog
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from face.facenet.model import Resnet34Triplet
from PySide6.QtCore import Qt


# 增强的数据集类
class EnhancedFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, target_cls, transform=None, use_hard_triplets=True, model=None):
        """
        Args:
            root_dir (string): 包含所有人脸图像的目录，按身份组织
            target_cls (string): 目标注册的类别（学号/ID）
            transform (callable, optional): 应用于样本的变换
            use_hard_triplets (bool): 是否使用困难三元组挖掘
            model (nn.Module): 用于计算嵌入以进行困难三元组挖掘的模型
        """
        self.root_dir = root_dir
        self.target_cls = target_cls
        self.transform = transform
        self.samples = []  # (image_path, identity) 元组列表
        self.use_hard_triplets = use_hard_triplets
        self.model = model
        self.identity_to_idx = {}  # 身份到索引的映射

        # 加载数据集
        for identity in os.listdir(root_dir):
            identity_dir = os.path.join(root_dir, identity)
            if os.path.isdir(identity_dir):
                self.identity_to_idx[identity] = len(self.identity_to_idx)
                for img_name in os.listdir(identity_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity_dir, img_name)
                        self.samples.append((img_path, identity))

        # 根据指定策略生成三元组
        self.triplets = self.generate_triplets()

    def generate_triplets(self):
        """根据当前数据集状态生成三元组"""
        if self.use_hard_triplets and self.model is not None:
            return self.generate_smart_triplets()
        else:
            return self.generate_balanced_triplets()

    def generate_balanced_triplets(self, n_triplets=32):
        """生成均衡的三元组，确保目标类别有足够的代表性"""
        # 按身份将索引分组
        identity_dict = {}
        for idx, (_, identity) in enumerate(self.samples):
            if identity not in identity_dict:
                identity_dict[identity] = []
            identity_dict[identity].append(idx)

        # 确保至少有2个样本的有效身份
        valid_identities = [identity for identity, indices in identity_dict.items() if len(indices) >= 2]

        if len(valid_identities) < 2:
            raise ValueError("数据集必须包含至少2个身份，每个身份至少2个样本")

        triplets = []

        # 确保目标类别得到足够表示（如果存在）
        target_ratio = 0.7  # 70%的三元组使用目标类别作为锚点
        target_count = int(n_triplets * target_ratio) if self.target_cls in identity_dict else 0
        other_count = n_triplets - target_count

        # 为目标类别生成三元组
        if target_count > 0:
            target_indices = identity_dict[self.target_cls]
            other_identities = [i for i in valid_identities if i != self.target_cls]

            for _ in range(target_count):
                # 选择锚点和正例（相同身份的不同图像）
                anchor_idx, positive_idx = random.sample(target_indices, 2)

                # 选择负例身份和样本
                negative_identity = random.choice(other_identities)
                negative_idx = random.choice(identity_dict[negative_identity])

                triplets.append((anchor_idx, positive_idx, negative_idx))

        # 为其他类别生成三元组
        for _ in range(other_count):
            # 随机选择锚点身份
            anchor_identity = random.choice(valid_identities)

            # 选择锚点和正例
            if len(identity_dict[anchor_identity]) >= 2:
                anchor_idx, positive_idx = random.sample(identity_dict[anchor_identity], 2)

                # 选择负例身份
                negative_identities = [i for i in valid_identities if i != anchor_identity]
                negative_identity = random.choice(negative_identities)
                negative_idx = random.choice(identity_dict[negative_identity])

                triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def generate_smart_triplets(self):
        """基于当前模型状态生成智能三元组"""
        identity_dict = {}
        # 按身份将索引分组
        for idx, (_, identity) in enumerate(self.samples):
            if identity not in identity_dict:
                identity_dict[identity] = []
            identity_dict[identity].append(idx)

        # 确保至少有2个样本的有效身份
        valid_identities = [identity for identity, indices in identity_dict.items() if len(indices) >= 2]

        if len(valid_identities) < 2:
            raise ValueError("数据集必须包含至少2个身份，每个身份至少2个样本")

        # 将模型设置为评估模式并获取设备
        self.model.eval()
        device = next(self.model.parameters()).device

        # 计算所有样本的嵌入
        embeddings = {}
        with torch.no_grad():
            for idx, (img_path, _) in enumerate(self.samples):
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                img = img.unsqueeze(0).to(device)  # 添加批次维度
                embedding = self.model(img)
                embeddings[idx] = embedding.cpu()

        triplets = []

        # 重点关注目标类别
        if self.target_cls in identity_dict:
            # 每个目标类别样本生成多个三元组
            for anchor_idx in identity_dict[self.target_cls]:
                # 找到最困难的正例（同一身份，最大距离）
                anchor_embedding = embeddings[anchor_idx]
                positive_distances = []

                for pos_idx in identity_dict[self.target_cls]:
                    if pos_idx != anchor_idx:
                        pos_embedding = embeddings[pos_idx]
                        distance = torch.dist(anchor_embedding, pos_embedding).item()
                        positive_distances.append((pos_idx, distance))

                # 选择距离最远的正例
                if positive_distances:
                    positive_distances.sort(key=lambda x: x[1], reverse=True)
                    hardest_positive_idx = positive_distances[0][0]

                    # 找到半困难负例（距离比正例稍大但不太多）
                    negative_distances = []
                    for neg_identity in [i for i in valid_identities if i != self.target_cls]:
                        for neg_idx in identity_dict[neg_identity]:
                            neg_embedding = embeddings[neg_idx]
                            distance = torch.dist(anchor_embedding, neg_embedding).item()
                            negative_distances.append((neg_idx, distance))

                    negative_distances.sort(key=lambda x: x[1])

                    # 尝试找到半困难负例，如果没有则使用最困难的负例
                    pos_distance = positive_distances[0][1]
                    margin = 0.2  # 较小的间隔以确保合适的负例

                    semi_hard_neg_idx = None
                    for neg_idx, neg_distance in negative_distances:
                        if neg_distance > pos_distance and neg_distance < pos_distance + margin:
                            semi_hard_neg_idx = neg_idx
                            break

                    if semi_hard_neg_idx is None:
                        # 如果没有半困难负例，则使用最困难的负例
                        semi_hard_neg_idx = negative_distances[0][0]

                    triplets.append((anchor_idx, hardest_positive_idx, semi_hard_neg_idx))

        # 为其他类别生成一些三元组以维持多样性
        other_identities = [i for i in valid_identities if i != self.target_cls]
        for identity in other_identities:
            if len(identity_dict[identity]) >= 2:
                # 每个其他身份只生成一个三元组
                anchor_idx = random.choice(identity_dict[identity])

                # 找到适合的正例和负例
                other_samples = [i for i in identity_dict[identity] if i != anchor_idx]
                positive_idx = random.choice(other_samples)

                # 为负例选择不同的身份
                neg_identity = random.choice([i for i in valid_identities if i != identity])
                negative_idx = random.choice(identity_dict[neg_identity])

                triplets.append((anchor_idx, positive_idx, negative_idx))

        return triplets

    def get_image_by_idx(self, idx):
        """获取无任何变换的图像"""
        img_path, _ = self.samples[idx]
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]

        # 加载锚点图像
        anchor_path, _ = self.samples[anchor_idx]
        anchor_img = Image.open(anchor_path).convert('RGB')

        # 加载正例图像
        positive_path, _ = self.samples[positive_idx]
        positive_img = Image.open(positive_path).convert('RGB')

        # 加载负例图像
        negative_path, _ = self.samples[negative_idx]
        negative_img = Image.open(negative_path).convert('RGB')

        # 应用变换
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def refresh_triplets(self):
        """在训练期间刷新三元组"""
        self.triplets = self.generate_triplets()


def freeze_layers(model, freeze_ratio=0.7):
    """
    冻结模型一部分层的权重

    Args:
        model: 要冻结的模型
        freeze_ratio: 要冻结的层的比例（从前往后计算）
    """
    # 获取所有参数
    all_params = list(model.named_parameters())

    # 确定要冻结的参数数量
    num_to_freeze = int(len(all_params) * freeze_ratio)

    # 冻结前面的层
    print(f"冻结前 {num_to_freeze} 层参数（共 {len(all_params)} 层）")
    frozen_count = 0

    for name, param in all_params[:num_to_freeze]:
        param.requires_grad = False
        frozen_count += 1

    # 记录要训练的参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"已冻结 {frozen_count} 层参数")
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})")

    return model


# 优化的训练函数
def train(cls, freeze_ratio=0.7):
    img_path = '../images/train/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 增强的数据增强
    transform_train = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])

    # 创建目标类别的目录
    cls_dir = os.path.join(img_path, cls)
    if not os.path.exists(cls_dir):
        os.makedirs(cls_dir)

    # 生成额外的数据增强样本（如果图像很少）
    generate_augmented_samples(cls_dir)

    # 加载或初始化模型
    if os.path.exists('../face/facenet/weights/transferred_facenet_model.pt'):
        model = Resnet34Triplet(embedding_dimension=512)
        checkpoint = torch.load('../face/facenet/weights/transferred_facenet_model.pt',
                                map_location=device)
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load('../face/facenet/weights/model_resnet34_triplet.pt',
                                map_location=device)
        model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    # 冻结一部分层的权重
    model = freeze_layers(model, freeze_ratio)

    # 创建进度对话框
    progress_dialog = QProgressDialog("训练中...", "取消", 0, 100)
    progress_dialog.setWindowTitle("训练进度")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.show()

    # 添加权重衰减和更好的优化器
    # 只优化未冻结的参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=0.001, weight_decay=1e-4)

    # 使用更好的学习率计划，基于有效的验证损失
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                  verbose=True, min_lr=1e-5)

    # 三元组损失，略微调整裕度以提高稳定性
    criterion = nn.TripletMarginLoss(margin=0.3)

    # 训练循环参数
    num_epochs = 20  # 最大轮次
    patience = 3  # 早停参数
    batch_size = 8  # 较小的批次大小，更好地适应少量样本
    best_loss = float('inf')
    patience_counter = 0

    # 初始化数据集
    dataset = EnhancedFaceDataset(
        root_dir=img_path,
        target_cls=cls,
        transform=transform_train,
        use_hard_triplets=True,
        model=model
    )

    for epoch in range(num_epochs):
        # 刷新三元组
        if epoch > 0:
            dataset.model = model  # 更新模型引用
            dataset.refresh_triplets()  # 基于当前模型状态刷新三元组

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 训练一个轮次
        model.train()
        running_loss = 0.0
        batch_count = 0

        for i, (anchor, positive, negative) in enumerate(dataloader):
            # 将数据移至设备
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # 清零参数梯度
            optimizer.zero_grad()

            # 前向传播
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            # 计算三元组损失
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            batch_count += 1

            # 更新对话框进度
            progress = int((epoch * len(dataloader) + i + 1) / (num_epochs * len(dataloader)) * 100)
            progress_dialog.setValue(progress)

            if progress_dialog.wasCanceled():
                print("训练已取消")
                return

        # 计算平均损失
        epoch_loss = running_loss / batch_count if batch_count > 0 else float('inf')
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 更新学习率调度器
        scheduler.step(epoch_loss)

        # 早停检查
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            # 保存最佳模型
            torch.save(model.state_dict(), '../face/facenet/weights/transferred_facenet_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 确保进度对话框完成
    progress_dialog.setValue(100)
    print("训练完成")


def generate_augmented_samples(cls_dir, min_samples=6):
    """为类别生成额外的增强样本"""
    # 获取目录中的图像文件
    img_files = [f for f in os.listdir(cls_dir)
                 if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 如果样本数量已经足够，则不需要生成
    if len(img_files) >= min_samples:
        return

    # 定义增强变换
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5))
    ]

    # 为每个原始图像生成新的增强版本
    for img_file in img_files[:]:  # 使用原始列表副本
        img_path = os.path.join(cls_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        # 生成文件名不包含扩展名的部分
        file_base = os.path.splitext(img_file)[0]

        # 应用每个增强并保存新图像
        for i, transform in enumerate(augmentations):
            augmented_img = transform(img)
            aug_filename = f"{file_base}_aug{i}.jpg"
            aug_path = os.path.join(cls_dir, aug_filename)
            augmented_img.save(aug_path)

    print(f"为 {cls_dir} 生成了增强样本")


if __name__ == "__main__":
    # 添加可选参数控制冻结层的比例（默认冻结70%的层）
    train('2107090712', freeze_ratio=0.7)
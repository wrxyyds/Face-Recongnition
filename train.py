import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from face.facenet.model import Resnet34Triplet

# 自定义数据集类
class CustomFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.data.append((img_path, cls))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[label]  # 将字符串标签转换为数值标签
        return image, label


def generate_triplets_from_batch(labels):
    triplets = []
    labels = labels.cpu()
    label_to_indices = {}

    for idx, label in enumerate(labels):
        label = label.item()
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    for anchor_idx, anchor_label in enumerate(labels):
        anchor_label = anchor_label.item()
        positive_indices = label_to_indices[anchor_label]
        if len(positive_indices) < 2:
            continue  # 至少要两张图才构成三元组

        # 选择 positive，不等于 anchor_idx
        positive_idx = anchor_idx
        while positive_idx == anchor_idx:
            positive_idx = positive_indices[torch.randint(len(positive_indices), (1,)).item()]

        # 选择 negative，标签不同
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = labels[torch.randint(len(labels), (1,))].item()
        negative_idx = label_to_indices[negative_label][torch.randint(len(label_to_indices[negative_label]), (1,)).item()]

        triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets



# 迁移训练函数
# === 修复后的训练函数 ===
def transfer_train(model, dataloader, criterion, optimizer, num_epochs=10):
    device = torch.device("cpu")
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_triplets = 0

        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            triplets = generate_triplets_from_batch(batch_labels)

            for anchor_idx, positive_idx, negative_idx in triplets:
                anchor_img = batch_images[anchor_idx:anchor_idx+1]     # shape: [1, 3, H, W]
                positive_img = batch_images[positive_idx:positive_idx+1]
                negative_img = batch_images[negative_idx:negative_idx+1]

                optimizer.zero_grad()

                anchor_embedding = model(anchor_img)
                positive_embedding = model(positive_img)
                negative_embedding = model(negative_img)

                loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_triplets += 1

        if total_triplets > 0:
            epoch_loss = running_loss / total_triplets
        else:
            epoch_loss = 0.0

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model


# 主函数
def train():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((140, 140)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])

    # 加载自定义数据集
    dataset = CustomFaceDataset(root_dir='./images/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化预训练模型
    checkpoint = torch.load('./face/facenet/weights/model_resnet34_triplet.pt', map_location='cpu')
    model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # 定义损失函数和优化器
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 迁移训练模型
    trained_model = transfer_train(model, dataloader, criterion, optimizer, num_epochs=10)

    # 保存迁移训练后的模型
    torch.save(trained_model.state_dict(), 'face/facenet/weights/transferred_facenet_model.pt')

if __name__ == "__main__":
    train()
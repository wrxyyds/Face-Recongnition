import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PySide6.QtWidgets import QProgressDialog
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from face.facenet.model import Resnet34Triplet
from PySide6.QtCore import Qt


# 自定义数据集类
class CustomFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, cls, transform=None, use_hard_triplets=True, model=None):
        """
        Args:
            root_dir (string): Directory with all the face images organized by identity
            transform (callable, optional): Transform to be applied on a sample
            use_hard_triplets (bool): Whether to use hard triplet mining
            model (nn.Module): Model to compute embeddings for hard triplet mining
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # List of (image_path, identity) tuples
        self.use_hard_triplets = use_hard_triplets
        self.model = model

        # Load dataset
        for identity in os.listdir(root_dir):
            identity_dir = os.path.join(root_dir, identity)
            if os.path.isdir(identity_dir):
                for img_name in os.listdir(identity_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity_dir, img_name)
                        self.samples.append((img_path, identity))

        # Generate triplets based on specified strategy
        if use_hard_triplets and model is not None:
            self.triplets = generate_hard_triplets(model, self, cls)
        else:
            # Fallback to random triplet generation if model not provided
            self.triplets = generate_random_triplets(self)

    def get_image_by_idx(self, idx):
        """Get image without any transformation"""
        img_path, _ = self.samples[idx]
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]

        # Load anchor image
        anchor_path, _ = self.samples[anchor_idx]
        anchor_img = Image.open(anchor_path).convert('RGB')

        # Load positive image
        positive_path, _ = self.samples[positive_idx]
        positive_img = Image.open(positive_path).convert('RGB')

        # Load negative image
        negative_path, _ = self.samples[negative_idx]
        negative_img = Image.open(negative_path).convert('RGB')

        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

    def refresh_triplets(self, cls):
        """Refresh triplets during training"""
        if self.use_hard_triplets and self.model is not None:
            self.triplets = generate_hard_triplets(self.model, self, cls)
        else:
            self.triplets = generate_random_triplets(self)


def generate_random_triplets(dataset, batch_size=32):
    """Simple random triplet generation as fallback"""
    # Dictionary to store indices of samples by identity
    identity_dict = {}

    # Group indices by identity
    for idx, (_, identity) in enumerate(dataset.samples):
        if identity not in identity_dict:
            identity_dict[identity] = []
        identity_dict[identity].append(idx)

    # Filter identities with at least 2 samples (needed for anchor-positive pairs)
    valid_identities = [identity for identity, indices in identity_dict.items() if len(indices) >= 2]

    if len(valid_identities) < 2:
        raise ValueError("Dataset must contain at least 2 identities with 2+ samples each")

    triplets = []
    for _ in range(batch_size):
        # Randomly select an identity for anchor/positive
        anchor_identity = random.choice(valid_identities)

        # Select anchor and positive (different images, same identity)
        anchor_idx, positive_idx = random.sample(identity_dict[anchor_identity], 2)

        # Select a negative identity (different from anchor)
        negative_identities = [i for i in valid_identities if i != anchor_identity]
        negative_identity = random.choice(negative_identities)

        # Select a negative sample
        negative_idx = random.choice(identity_dict[negative_identity])

        triplets.append((anchor_idx, positive_idx, negative_idx))

    return triplets


def generate_hard_triplets(model, dataset, cls, margin=0.5):
    """
    Generate triplets using hard mining strategy:
    - Use specified identity as anchor
    - Find hardest positive sample (same identity but most dissimilar to anchor)
    - Find hardest negative sample (different identity but most similar to anchor)

    Args:
        model: The face embedding model used to compute distances
        dataset: Dataset containing face images with identity labels
        batch_size: Number of triplets to generate
        margin: Margin for triplet loss

    Returns:
        List of triplets (anchor_idx, positive_idx, negative_idx)
    """
    # Dictionary to store indices of samples by identity
    identity_dict = {}

    # Group indices by identity
    for idx, (_, identity) in enumerate(dataset.samples):
        if identity not in identity_dict:
            identity_dict[identity] = []
        identity_dict[identity].append(idx)

    # Filter identities with at least 2 samples (needed for anchor-positive pairs)
    valid_identities = [identity for identity, indices in identity_dict.items() if len(indices) >= 2]

    if len(valid_identities) < 2:
        raise ValueError("Dataset must contain at least 2 identities with 2+ samples each")

    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get device model is on

    triplets = []
    for _ in range(len(identity_dict[cls])):
        # Randomly select an identity for anchor
        anchor_identity = cls
        anchor_indices = identity_dict[anchor_identity]

        # Randomly select an anchor sample from this identity
        anchor_idx = random.choice(anchor_indices)

        # Get anchor embedding
        anchor_img = dataset.get_image_by_idx(anchor_idx)
        if dataset.transform:
            anchor_img = dataset.transform(anchor_img)
        anchor_img = anchor_img.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            anchor_embedding = model(anchor_img)

        # Find hardest positive (same identity, maximum distance)
        max_pos_distance = -1
        hardest_positive_idx = None

        for pos_idx in anchor_indices:
            if pos_idx == anchor_idx:  # Skip the anchor itself
                continue

            pos_img = dataset.get_image_by_idx(pos_idx)
            if dataset.transform:
                pos_img = dataset.transform(pos_img)
            pos_img = pos_img.unsqueeze(0).to(device)

            with torch.no_grad():
                pos_embedding = model(pos_img)

            # Calculate distance
            distance = torch.dist(anchor_embedding, pos_embedding).item()

            if distance > max_pos_distance:
                max_pos_distance = distance
                hardest_positive_idx = pos_idx

        # Find hardest negative (different identity, minimum distance)
        min_neg_distance = float('inf')
        hardest_negative_idx = None

        other_identities = [i for i in valid_identities if i != anchor_identity]

        for neg_identity in other_identities:
            for neg_idx in identity_dict[neg_identity]:
                neg_img = dataset.get_image_by_idx(neg_idx)
                if dataset.transform:
                    neg_img = dataset.transform(neg_img)
                neg_img = neg_img.unsqueeze(0).to(device)

                with torch.no_grad():
                    neg_embedding = model(neg_img)

                # Calculate distance
                distance = torch.dist(anchor_embedding, neg_embedding).item()

                if distance < min_neg_distance:
                    min_neg_distance = distance
                    hardest_negative_idx = neg_idx

        # Check if this is a semi-hard negative sample
        # We want neg_distance > pos_distance but not by too much (less than margin)
        if min_neg_distance > max_pos_distance and min_neg_distance < max_pos_distance + margin:
            triplets.append((anchor_idx, hardest_positive_idx, hardest_negative_idx))
        else:
            # If no semi-hard negative, just use the hardest negative
            triplets.append((anchor_idx, hardest_positive_idx, hardest_negative_idx))

    return triplets


# 一次epoch的训练步骤
def train(cls):
    img_path = '../images/train/'
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((140, 140)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])

    # Load or initialize model
    if os.path.exists('../face/facenet/weights/transferred_facenet_model.pt'):
        model = Resnet34Triplet(embedding_dimension=512)
        checkpoint = torch.load('../face/facenet/weights/transferred_facenet_model.pt')
        model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load('../face/facenet/weights/model_resnet34_triplet.pt', map_location=device)
        model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
        model.load_state_dict(checkpoint['model_state_dict'])
    """
    checkpoint = torch.load('../face/facenet/weights/model_resnet34_triplet.pt', map_location=device)
    model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
    model.load_state_dict(checkpoint['model_state_dict'])
    """

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.TripletMarginLoss(margin=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Create progress dialog

    # First initialize dataset with model to generate hard triplets
    dataset = CustomFaceDataset(root_dir=img_path, transform=transform, use_hard_triplets=True, model=model, cls=cls)

    progress_dialog = QProgressDialog("Training in progress...", "Cancel", 0, num_epochs)
    progress_dialog.setWindowTitle("Training Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        # Create new dataloader with refreshed triplets
        if epoch > 0:
            dataset.model = model  # Update model reference
            dataset.refresh_triplets(cls)  # Refresh triplets based on current model state

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Train one epoch
        model.train()
        running_loss = 0.0

        for i, (anchor, positive, negative) in enumerate(dataloader):
            # Move data to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            # Compute triplet loss
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            print(loss)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            if i % 5 == 4:  # Print every 5 mini-batches
                progress_str = f'Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}, Loss: {running_loss / 5:.4f}'
                print(progress_str)
                running_loss = 0.0

        # Step learning rate scheduler
        scheduler.step()

        # Update progress bar

        # Save model after each epoch
        torch.save(model.state_dict(), '../face/facenet/weights/transferred_facenet_model.pt')
        progress_dialog.setValue(epoch + 1)
        if progress_dialog.wasCanceled():
            break


if __name__ == "__main__":
    train('2107090712')

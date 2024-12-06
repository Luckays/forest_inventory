# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from deep_learning.models.pointnet2_sem_seg_msg import get_model, get_loss  # Importing PointNet++ segmentation model


# Step 1: Create a Custom Dataset
class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # Load the data from HDF5 file
        self.data = []
        self.labels = []
        with h5py.File(data_path, 'r') as h5f:
            blocks_grp = h5f['blocks']
            for block_name in blocks_grp:
                block_grp = blocks_grp[block_name]
                points = block_grp['points'][:, :4]  # XYZ + intensity
                labels = block_grp['points'][:, -1]  # Classification (last column)
                self.data.append(points)
                self.labels.append(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_block = self.data[idx]  # Shape: (1024, 4) - (x, y, z, intensity)
        labels = self.labels[idx]  # Shape: (1024,) - class labels for each point
        return torch.tensor(point_block, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# Step 2: Define the Training Function
def train_pointnet(data_path, epochs=10, batch_size=16, learning_rate=0.001):
    # Create dataset and dataloader
    dataset = PointCloudDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, loss function, and optimizer
    num_classes = 100  # Assuming 100 classes for individual tree segmentation
    model = get_model(num_classes).cuda()
    criterion = get_loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (points, labels) in enumerate(dataloader):
            points = points.permute(0, 2, 1)  # Adjusting shape to (batch_size, num_features, num_points) for PointNet++
            points, labels = points.cuda(), labels.cuda()

            # Forward pass
            optimizer.zero_grad()
            pred, _ = model(points)
            loss = criterion(pred, labels, None, None)  # Adjusted for segmentation loss

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

    print('Training completed')


# Example usage
if __name__ == "__main__":
    data_path = r"C:\Users\lukas\Desktop\pointcloud_blocks3.h5"  # Path to your HDF5 dataset
    train_pointnet(data_path, epochs=20, batch_size=8, learning_rate=0.001)

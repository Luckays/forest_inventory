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
            if 'voxels' not in h5f:
                raise ValueError("Invalid file format: 'voxels' group not found.")

            voxels_grp = h5f['voxels']
            for voxel_name in voxels_grp:
                voxel_grp = voxels_grp[voxel_name]

                if 'blocks' not in voxel_grp:
                    continue

                blocks_grp = voxel_grp['blocks']
                for block_name in blocks_grp:
                    block_data = blocks_grp[block_name][()]
                    if block_data.shape[1] < 5:  # Ensure at least 5 columns (XYZ, intensity, label)
                        raise ValueError(f"Block {block_name} has insufficient columns.")

                    points = block_data[:, :4]  # XYZ + intensity
                    labels = block_data[:, 4]  # Classification (5th column)

                    self.data.append(points)
                    self.labels.append(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_block = self.data[idx]  # Shape: (1024, 4) - (x, y, z, intensity)
        labels = self.labels[idx]  # Shape: (1024,) - class labels for each point
        return torch.tensor(point_block, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# Step 2: Define the Training Function
def train_pointnet(data_path, epochs, batch_size, learning_rate):
    # Create dataset and dataloader
    dataset = PointCloudDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the model, loss function, and optimizer
    num_classes = int(max([label.max().item() for label in dataset.labels])) + 1
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


    print('Training completed')


# Example usage
if __name__ == "__main__":
    data_path = r"C:\Users\lukas\Desktop\Test\H5\pointcloud_big.h5"  # Path to your HDF5 dataset
    train_pointnet(data_path, epochs=20, batch_size=16, learning_rate=0.001)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Step 1: Create a Custom Dataset
# class PointCloudDataset(Dataset):
#     def __init__(self, data_path):
#         super().__init__()
#         # Load the data from HDF5 file
#         self.data = []
#         self.max_label = 0
#         with h5py.File(data_path, 'r') as h5f:
#             blocks_grp = h5f['blocks']
#             for block_name in blocks_grp:
#                 block_grp = blocks_grp[block_name]
#                 points = block_grp['points'][:]
#                 self.data.append(points)
#                 label = points[0, -1]
#                 if label > self.max_label:
#                     self.max_label = label
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         point_block = self.data[idx]  # Shape: (1024, 4) - (x, y, z, intensity)
#         points = point_block[:, :4]  # XYZ + intensity
#         label = point_block[0, -1]  # Assuming all points in the block have the same class label
#         return points, label
#
# # Step 2: Define the PointNet++ Model (using a basic architecture for simplicity)
# class PointNetPlusPlus(nn.Module):
#     def __init__(self, num_classes):
#         super(PointNetPlusPlus, self).__init__()
#         # Simplified PointNet++ layers for feature learning
#         self.sa1 = nn.Sequential(
#             nn.Conv1d(4, 64, 1),
#             nn.ReLU(),
#             nn.Conv1d(64, 128, 1),
#             nn.ReLU(),
#             nn.Conv1d(128, 256, 1),
#             nn.ReLU(),
#         )
#         self.fc1 = nn.Linear(256, 128)
#         self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = x.transpose(2, 1)  # Input shape: (batch_size, num_points, num_features) -> (batch_size, num_features, num_points)
#         x = self.sa1(x)
#         x, _ = torch.max(x, 2)  # Global feature (max pooling)
#         x = x.view(-1, 256)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# # Step 3: Training Loop
# if __name__ == "__main__":
#     # Dataset and DataLoader
#     dataset = PointCloudDataset(r"C:\Users\lukas\Desktop\pointcloud_blocks.h5")
#     num_classes = int(dataset.max_label) + 1  # Set num_classes based on the maximum label value in the dataset
#
#     # Hyperparameters
#     batch_size = 16
#     num_epochs = 100
#     learning_rate = 0.00001
#
#     # Dataset and DataLoader
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     # Model, Loss, and Optimizer
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = PointNetPlusPlus(num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     # Training Loop
#     loss_history = []
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for i, (points, labels) in enumerate(dataloader):
#             points, labels = points.to(device).float(), labels.to(device).long()
#             optimizer.zero_grad()
#
#             # Forward pass
#             outputs = model(points)
#             loss = criterion(outputs, labels)
#
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         epoch_loss = running_loss / len(dataloader)
#         loss_history.append(epoch_loss)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#
#     # Plot the loss function
#     plt.figure()
#     plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss over Epochs')
#     plt.grid()
#     plt.show()
#
#     print("Training Finished!")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a Custom Dataset
class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # Load the data from HDF5 file
        self.data = []
        self.max_label = 0
        with h5py.File(data_path, 'r') as h5f:
            blocks_grp = h5f['blocks']
            for block_name in blocks_grp:
                block_grp = blocks_grp[block_name]
                points = block_grp['points'][:]
                self.data.append(points)
                label = points[0, -1]
                if label > self.max_label:
                    self.max_label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_block = self.data[idx]  # Shape: (1024, 4) - (x, y, z, intensity)
        points = point_block[:, :4]  # XYZ + intensity
        label = point_block[0, -1]  # Assuming all points in the block have the same class label
        return points, label

# Step 2: Define the PointNet++ Model (using a basic architecture for simplicity)
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        # Simplified PointNet++ layers for feature learning
        self.sa1 = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1)  # Input shape: (batch_size, num_points, num_features) -> (batch_size, num_features, num_points)
        x = self.sa1(x)
        x, _ = torch.max(x, 2)  # Global feature (max pooling)
        x = x.view(-1, 256)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Training Loop
if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = PointCloudDataset(r"C:\Users\lukas\Desktop\pointcloud_blocks.h5")
    num_classes = int(dataset.max_label) + 1  # Set num_classes based on the maximum label value in the dataset

    # Hyperparameters
    batch_size = 16
    num_epochs = 100
    learning_rate = 0.00001

    # Dataset and DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetPlusPlus(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    loss_history = []
    accuracy_history = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (points, labels) in enumerate(dataloader):
            points, labels = points.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(points)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Plot the loss function and accuracy
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_epochs + 1), accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy over Epochss')
    plt.grid()

    plt.tight_layout()
    plt.show()

    print("Training Finished!")

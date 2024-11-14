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
        self.labels = []
        with h5py.File(data_path, 'r') as h5f:
            blocks_grp = h5f['blocks']
            for block_name in blocks_grp:
                block_grp = blocks_grp[block_name]
                points = block_grp['points'][:, :3]  # XYZ (removed intensity)
                labels = block_grp['points'][:, -1]  # Classification (last column)
                self.data.append(points)
                self.labels.append(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_block = self.data[idx]  # Shape: (1024, 3) - (x, y, z)
        labels = self.labels[idx]  # Shape: (1024,) - class labels for each point
        return torch.tensor(point_block, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# Step 2: Define the PointNet++ Model (using a basic architecture for simplicity)
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes):
        super(PointNetPlusPlus, self).__init__()
        # Simplified PointNet++ layers for feature learning
        self.sa1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),  # First convolutional layer, input features are XYZ (3 channels)
            nn.ReLU(),  # Activation function
            nn.Conv1d(64, 128, 1),  # Second convolutional layer
            nn.ReLU(),  # Activation function
            nn.Conv1d(128, 256, 1),  # Third convolutional layer
            nn.ReLU(),  # Activation function
        )
        self.fc1 = nn.Conv1d(256, 128, 1)  # Fully connected layer to reduce features to 128 for each point
        self.fc2 = nn.Conv1d(128, num_classes, 1)  # Fully connected layer to output number of classes for each point

    def forward(self, x):
        x = x.transpose(2, 1)  # Transpose input shape to (batch_size, num_features, num_points)
        x = self.sa1(x)  # Apply convolutional layers
        x = self.fc1(x)  # Apply fully connected layer
        x = self.fc2(x)  # Output layer for each point (batch_size, num_classes, num_points)
        x = x.transpose(2, 1)  # Output shape: (batch_size, num_points, num_classes)
        return x

# Step 3: Training Loop
if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = PointCloudDataset(r"C:\Users\lukas\Desktop\pointcloud_blocks3.h5")
    num_classes = int(max([label.max().item() for label in dataset.labels])) + 1  # Set num_classes based on the maximum label value in the dataset

    # Hyperparameters
    batch_size = 16
    num_epochs = 200
    learning_rate = 0.00001

    # Dataset and DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetPlusPlus(num_classes).to(device)  # Initialize the model and move it to the device
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

    # Training Loop
    loss_history = []
    accuracy_history = []
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (points, labels) in enumerate(dataloader):
            points, labels = points.to(device).float(), labels.to(device).long()  # Move data to the device
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(points)  # Shape: (batch_size, num_points, num_classes)
            outputs = outputs.reshape(-1, num_classes)  # Flatten outputs to (batch_size * num_points, num_classes)
            labels = labels.view(-1)  # Flatten labels to (batch_size * num_points)

            # Loss calculation
            loss = criterion(outputs, labels)  # Compute the loss

            # Backward pass and optimization
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model parameters

            running_loss += loss.item()  # Accumulate the loss

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Count correct predictions

        epoch_loss = running_loss / len(dataloader)  # Average loss for the epoch
        epoch_accuracy = 100 * correct / total  # Accuracy for the epoch
        loss_history.append(epoch_loss)  # Store loss history
        accuracy_history.append(epoch_accuracy)  # Store accuracy history
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
    plt.title('Training Accuracy over Epochs')
    plt.grid()

    plt.tight_layout()
    plt.show()

    print("Training Finished!")
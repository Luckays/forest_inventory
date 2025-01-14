import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.metrics import classification_report
import torch.nn.functional as F
from deep_learning.models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation

# Directory structure and configs
CONFIG_PATH = "config.yaml"


class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)


# Dataset loader
class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # Load the data from HDF5 file
        self.data = []
        self.labels = []
        block = []
        with h5py.File(data_path, 'r') as h5f:
            if 'voxels' not in h5f:
                raise ValueError("Invalid file format: 'voxels' group not found.")

            voxels_grp = h5f['voxels']
            for voxel_name in voxels_grp:
                voxel_grp = voxels_grp[voxel_name]
                print(voxel_name)
                if 'blocks' not in voxel_grp:
                    continue

                blocks_grp = voxel_grp['blocks']
                for block_name in blocks_grp:
                    print(block_name)
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
        point_block = self.data[idx]
        labels = self.labels[idx]
        return torch.tensor(point_block, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# Model definition
class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 4, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, points):
        l0_points = points
        l0_xyz = points[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


# Loss definition
class PointNetLoss(nn.Module):
    def __init__(self):
        super(PointNetLoss, self).__init__()

    def forward(self, pred, target, weight):
        pred = pred.permute(0, 2, 1)
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

# Training script
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        dataset = PointCloudDataset(self.config.get('data_path'))
        dataloader = DataLoader(dataset, batch_size=self.config.get('batch_size'), shuffle=True, num_workers=0)


        num_classes = int(max(label.max().item() for label in dataset.labels) + 1)
        model = PointNetSegmentation(num_classes).to(self.device)
        criterion = PointNetLoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.get('learning_rate'))

        class_counts = np.bincount(np.concatenate(dataset.labels).astype(int), minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-8)  # Prevent division by zero
        class_weights[class_counts == 0] = 0  # Ensure weights for absent labels are zero
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        print("classes done")

        for epoch in range(self.config.get('epochs')):
            print("epoch: " + str(epoch))
            model.train()
            print("model")
            running_loss = 0.0

            print(dataloader)


            for i, (points, labels) in enumerate(dataloader):
                print("Point: " + str(points.shape))
                points, labels = points.permute(0, 2, 1).to(self.device), labels.to(self.device)
                print("Permute: " + str(points.shape))
                optimizer.zero_grad()
                pred = model(points)
                loss = F.nll_loss(pred.permute(0, 2, 1), labels, weight=class_weights)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{self.config.get('epochs')}, Loss: {running_loss / len(dataloader)}")

        torch.save(model.state_dict(), self.config.get('model_save_path'))

    def test(self):
        dataset = PointCloudDataset(self.config.get('test_data_path'))
        dataloader = DataLoader(dataset, batch_size=self.config.get('batch_size'), shuffle=False, num_workers=4)

        num_classes = int(max(label.max().item() for label in dataset.labels) + 1)
        model = PointNetSegmentation(num_classes).to(self.device)
        model.load_state_dict(torch.load(self.config.get('model_save_path'), weights_only=True))

        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for points, labels in dataloader:
                points, labels = points.permute(0, 2, 1).to(self.device), labels.to(self.device)
                preds = model(points).argmax(dim=2).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(np.concatenate(all_labels), np.concatenate(all_preds))
        print("Classification Report:")
        print(report)

        # Save report to a text file
        with open("classification_report.txt", "w") as f:
            f.write("Classification Report:\n")
            f.write(report)

if __name__ == "__main__":
    config = Config(CONFIG_PATH)
    trainer = Trainer(config)
    # trainer.train()
    trainer.test()

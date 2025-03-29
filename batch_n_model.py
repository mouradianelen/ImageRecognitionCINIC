import torch.nn as nn
import torch.nn.functional as F
import torch

class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with increased depth
        self.conv1 = nn.Conv2d(
            3, 16, 3, padding=1
        )  # Input: 3 channels, Output: 16 channels, 3x3 kernel, padding=1
        self.bn1 = nn.BatchNorm2d(16)  # added bn
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 kernel
        self.conv2 = nn.Conv2d(
            16, 32, 3, padding=1
        )  # Input: 16 channels, Output: 32 channels, 3x3 kernel, padding=1
        self.bn2 = nn.BatchNorm2d(32)  # added bn
        self.conv3 = nn.Conv2d(
            32, 64, 3, padding=1
        )  # Input: 32 channels, Output: 64 channels, 3x3 kernel, padding=1
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 4 * 4, 120)  # Adjusted for new dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Output: 16 channels, 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Output: 32 channels, 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Output: 64 channels, 4x4
        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function for the final layer

        return x


net = BatchNormNet()
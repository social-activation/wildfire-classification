import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Convolutional encoder
        self.conv1 = nn.Conv2d(3, 4, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(4, 8, 5) # 6 input channels, 16 output channels, 5x5 kernel

        self.dropout = nn.Dropout(0.3)

        # Fully connected layers / Dense block
        # self.fc1 = nn.Linear(8 * 29*29, 32)   # For 128,128
        self.fc1 = nn.Linear(1352, 32)          # For 64,64
        # self.fc1 = nn.Linear(200, 32)         # For 32,32
        self.fc2 = nn.Linear(32, 16)         # 120 inputs, 84 outputs
        self.fc3 = nn.Linear(16, 1)          # 84 inputs, 10 outputs (number of classes)

    def forward(self, x):
        # Convolutional block
        x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool
        x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2)) # Convolution -> Sigmoid -> Avg Pool

        # Flattening
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function here, will use CrossEntropyLoss later
        x = torch.sigmoid(x)
        return x

class SmallLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.pool = nn.AvgPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))  # (B, 4, 30, 30) → (B, 4, 15, 15)
        x = self.pool(self.relu2(self.conv2(x)))  # (B, 8, 11, 11) → (B, 8, 5, 5)
        x = self.global_pool(x)               # → (B, 8, 1, 1)
        x = x.view(x.size(0), -1)             # → (B, 8)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

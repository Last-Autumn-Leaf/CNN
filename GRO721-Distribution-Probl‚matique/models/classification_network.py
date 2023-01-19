import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationNetwork(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ClassificationNetwork, self).__init__()
        #ALEX NET
        self.n_params = n_classes
        self.kernels_size = 2
        params = [8, 16, 128]
        num_kernels = 2

        self.conv_1 = nn.Conv2d(in_channels, params[0], 11, stride=4, padding=2)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(3, stride=2, padding=0)
        self.conv_2 = nn.Conv2d(params[0], params[1], num_kernels, padding=2, stride=1)
        self.relu_2 = nn.ReLU()
        self.maxpool_5 = nn.MaxPool2d(num_kernels, padding=0, stride=2)
        self.flat_5 = nn.Flatten()

        self.outLayer = nn.Sequential(
            nn.Linear(256, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.maxpool_5(x)
        x = self.flat_5(x)

        output = self.outLayer(x)
        return output
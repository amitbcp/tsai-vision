import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dropout_value=0):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # In: 32x32x3 | Out: 32x32x32 | RF: 3x3
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # In: 32x32x32 | Out: 32x32x32 | RF: 5x5
        )
        # In: 32x32x32 | Out: 16x16x32 | RF: 6x6
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # In: 16x16x32 | Out: 16x16x64 | RF: 10x10
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # In: 16x16x64 | Out: 16x16x64 | RF: 14x14
        )
        # In: 16x16x64 | Out: 8x8x64 | RF:16x16
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(
                3, 3), padding=1, groups=64, bias=False),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # In: 8x8x64 | Out: 8x8x64 | RF: 24x24
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # In: 8x8x64 | Out: 8x8x64 | RF: 32x32
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # In: 8x8x64 | Out: 4x4x64 | RF: 36x36
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(
                3, 3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # In: 4x4x64 | Out: 4x4x128 | RF: 68x68
            nn.Dropout(dropout_value),

            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),  # In: 4x4x128 | Out: 4x4x128 | RF: 84x84
        )
        # In: 4x4x128 | Out: 1x1x128 | RF: 108x108
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
            # nn.ReLU() NEVER!
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.layer5(x)
        return x


net = Net()


class Model_loader:

    def models(device):

        model = Net().to(device)
        print(summary(model, input_size=(3, 32, 32)))
        return model

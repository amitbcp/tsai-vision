import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def DepthSep2d(in_channels: int, out_channels: int, kernel_size=3):
            depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels)
            point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

            return nn.Sequential(depth_conv, point_conv)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            DepthSep2d(in_channels=32, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True)
            
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.gap(x)
        print(x.size())
        print(x.numel())
        x = x.reshape(-1, 128)
        return x


net = Net()


class Model_loader:

    def models(device):

        model = Net().to(device)
        print(summary(model, input_size=(3, 32, 32)))
        return model

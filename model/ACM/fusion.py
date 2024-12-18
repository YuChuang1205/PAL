import torch.nn as nn
from torch.nn import BatchNorm2d
import torch
from torch.nn import GroupNorm
class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)


        self.topdown = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        # nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.GroupNorm(1, self.bottleneck_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels,  kernel_size=1, stride=1,padding=0),
        #nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.GroupNorm(1, self.channels),
        nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
        nn.Conv2d(in_channels=self.channels,out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        #nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.GroupNorm(1, self.bottleneck_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels, kernel_size=1, stride=1,padding=0),
        #nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.GroupNorm(1, self.channels),
        nn.Sigmoid()
        )

        self.post = nn.Sequential(
        nn.Conv2d(in_channels=channels,out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
        #nn.BatchNorm2d(channels,momentum=0.9),
        nn.GroupNorm(1, self.channels),
        nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * torch.mul(xl, topdown_wei) + 2 * torch.mul(xh, bottomup_wei)
        xs = self.post(xs)
        return xs

# from mxnet.gluon.block import HybridBlock
# topdown = HybridBlock(prefix='topdown')
# print(topdown)
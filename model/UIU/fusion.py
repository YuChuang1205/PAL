import torch
import torch.nn as nn

class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            #nn.BatchNorm2d(out_channels),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(True),
        )##512

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            #nn.BatchNorm2d(self.bottleneck_channels),
            nn.GroupNorm(1, self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            #nn.BatchNorm2d(self.out_channels),
            nn.GroupNorm(1, out_channels),
            nn.Sigmoid(),
        )#512

        ##############add spatial attention ###Cross UtU############
        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            #nn.BatchNorm2d(self.bottleneck_channels),
            nn.GroupNorm(1, self.bottleneck_channels),
            nn.ReLU(True),
            # nn.Sigmoid(),

            SpatialAttention(kernel_size=3),
            # nn.Conv2d(self.bottleneck_channels, 2, 3, 1, 0),
            # nn.Conv2d(2, 1, 1, 1, 0),
            #nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            #nn.BatchNorm2d(self.out_channels),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(True),
        )#512

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl * topdown_wei)
        xs1 = 2 * xl * topdown_wei  #1
        out1 = self.post(xs1)

        xs2 = 2 * xh * bottomup_wei    #1
        out2 = self.post(xs2)
        return out1,out2

        ##############################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

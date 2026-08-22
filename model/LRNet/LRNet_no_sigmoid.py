import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, groups=groups)
        self.depthwise.apply(weights_init)
        self.pointwise.apply(weights_init)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Resnet2(nn.Module):
    def __init__(self, in_channel, out_channel, groups=1):
        super(Resnet2, self).__init__()
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ECA(out_channel),
        )
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2, groups=groups),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

        self.eca = ECA(out_channel)
        #  网络初始化
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        identity = self.layer2(identity)
        identity = self.eca(identity)
        out += identity
        return self.relu(out)


class Stage(nn.Module):
    def __init__(self):
        super(Stage, self).__init__()
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv(3, 4, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            ECA(4),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv(4, 4, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            ECA(4)
        )

        self.resnet2_1 = Resnet2(4, 8, groups=1)
        self.resnet3_1 = Resnet2(12, 16, groups=1)
        self.resnet4_1 = Resnet2(20, 32, groups=1)
        self.resnet5_1 = Resnet2(36, 64, groups=1)

        self.layer1.apply(weights_init)

        self.maxpooling_16 = nn.MaxPool2d(kernel_size=(16,16),stride=(16,16))
        self.maxpooling_8 = nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8))
        self.maxpooling_4 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.maxpooling_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


    def forward(self, x):
        outs = []
        out_1 = self.layer1(x)
        outs.append(out_1)

        out_2 = self.resnet2_1(out_1)
        pooling_1_to_2 = self.maxpooling_2(out_1)
        out_2 = torch.cat((out_2, pooling_1_to_2), dim=1)
        outs.append(out_2)

        out_3 = self.resnet3_1(out_2)
        pooling_1_to_3 = self.maxpooling_4(out_1)
        out_3 = torch.cat((out_3, pooling_1_to_3), dim=1)
        outs.append(out_3)

        out_4 = self.resnet4_1(out_3)
        pooling_1_to_4 = self.maxpooling_8(out_1)
        out_4 = torch.cat((out_4, pooling_1_to_4), dim=1)
        outs.append(out_4)

        out_5 = self.resnet5_1(out_4)
        pooling_1_to_5 = self.maxpooling_16(out_1)
        out_5 = torch.cat((out_5, pooling_1_to_5), dim=1)
        outs.append(out_5)
        return outs



class LCL(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(LCL, self).__init__()
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ECA(out_channels),
        )
        self.layer1.apply(weights_init)


    def forward(self, x):
        out = self.layer1(x)
        return out


class Sbam(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super(Sbam, self).__init__()
        self.hl_layer = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.ll_layer = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=1, groups=groups)
        self.sigmoid = nn.Sigmoid()

        self.hl_layer.apply(weights_init)
        self.eca = ECA(out_channels)


    def forward(self, hl, ll):
        hl = self.hl_layer(hl)
        ll_1 = ll
        ll = self.ll_layer(ll)
        ll = self.sigmoid(ll)
        hl_1 = hl * ll
        out = ll_1 + hl_1
        out = self.eca(out)
        return out


class LRNet_No_Sigmoid(nn.Module):
    def __init__(self):
        super(LRNet_No_Sigmoid, self).__init__()
        self.stage = Stage()
        self.lcl5 = LCL(68, 64, groups=1)
        self.lcl4 = LCL(36, 32, groups=1)
        self.lcl3 = LCL(20, 16, groups=1)
        self.lcl2 = LCL(12, 8, groups=1)
        self.lcl1 = LCL(4, 4, groups=1)
        self.sbam4 = Sbam(64, 32, groups=1)
        self.sbam3 = Sbam(32, 16, groups=1)
        self.sbam2 = Sbam(16, 8, groups=1)
        self.sbam1 = Sbam(8, 4, groups=1)

        self.layer = nn.Sequential(
            DepthwiseSeparableConv(4, 4, kernel_size=1, groups=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            ECA(4),
            #nn.UpsamplingNearest2d(scale_factor=2),
            DepthwiseSeparableConv(4, 1, kernel_size=1, groups=1),
        )
        self.layer.apply(weights_init)

    def forward(self, x):
        outs = self.stage(x)
        out5 = self.lcl5(outs[4])
        out4 = self.lcl4(outs[3])
        out3 = self.lcl3(outs[2])
        out2 = self.lcl2(outs[1])
        out1 = self.lcl1(outs[0])
        out4_2 = self.sbam4(out5, out4)
        out3_2 = self.sbam3(out4_2, out3)
        out2_2 = self.sbam2(out3_2, out2)
        out1_2 = self.sbam1(out2_2, out1)
        out = self.layer(out1_2)
        return out

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    return m

















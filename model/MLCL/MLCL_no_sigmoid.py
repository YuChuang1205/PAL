"""
@author: yuchuang,zhaojinmiao
@time: 
@desc: 这个版本是MLCL-Net的基础版本（论文一致）。即每阶段用了3个block      paper:"Infrared small target detection based on multiscale local contrast learning networks"
"""
import torch
import torch.nn as nn

class Resnet1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer.apply(weights_init)

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out += identity
        return self.relu(out)


class Resnet2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        identity = self.layer2(identity)
        out += identity
        return self.relu(out)


class Stage(nn.Module):
    def __init__(self):
        super(Stage, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.resnet1_1 = Resnet1(in_channel=16, out_channel=16)
        self.resnet1_2 = Resnet1(in_channel=16, out_channel=16)
        self.resnet1_3 = Resnet1(in_channel=16, out_channel=16)
        self.resnet2_1 = Resnet2(in_channel=16, out_channel=32)
        self.resnet2_2 = Resnet1(in_channel=32, out_channel=32)
        self.resnet2_3 = Resnet1(in_channel=32, out_channel=32)
        self.resnet3_1 = Resnet2(in_channel=32, out_channel=64)
        self.resnet3_2 = Resnet1(in_channel=64, out_channel=64)
        self.resnet3_3 = Resnet1(in_channel=64, out_channel=64)

    def forward(self, x):
        outs = []
        out = self.layer1(x)
        out = self.resnet1_1(out)
        out = self.resnet1_2(out)
        out = self.resnet1_3(out)
        outs.append(out)
        out = self.resnet2_1(out)
        out = self.resnet2_2(out)
        out = self.resnet2_3(out)
        outs.append(out)
        out = self.resnet3_1(out)
        out = self.resnet3_2(out)
        out = self.resnet3_3(out)
        outs.append(out)
        return outs


class MLCL(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLCL, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=3, stride=1, dilation=3),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=5, stride=1, dilation=5),
            nn.ReLU(inplace=True)
        )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, padding=2, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=7, stride=1,dilation=7),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = nn.Conv2d(in_channels=out_channel * 3, out_channels=out_channel, kernel_size=1)
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)
        self.layer3.apply(weights_init)
        self.conv.apply(weights_init)
    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x
        out1 = self.layer1(x1)
        out2 = self.layer2(x2)
        out3 = self.layer3(x3)
        outs = torch.cat((out1, out2, out3), dim=1)
        return self.conv(outs)


class MLCL_No_Sigmoid(nn.Module):
    def __init__(self):
        super(MLCL_No_Sigmoid, self).__init__()
        self.stage = Stage()
        self.mlcl3 = MLCL(64, 64)
        self.mlcl2 = MLCL(32, 32)
        self.mlcl1 = MLCL(16, 16)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            #nn.Sigmoid()
        )

    def forward(self, x):
        outs = self.stage(x)
        out3 = self.mlcl3(outs[2])
        out2 = self.mlcl2(outs[1])
        out1 = self.mlcl1(outs[0])
        out3 = self.conv3(out3)  # 128*128 64
        out3 = self.up3(out3)    # 256*256 64
        out2 = self.conv2(out2)  # 256*256 64
        out = out3 + out2
        out = self.up2(out)
        out1 = self.conv1(out1)
        out = out + out1
        out = self.layer(out)
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
    return

if __name__ == '__main__':
    model = MLCL_No_Sigmoid()
    x = torch.rand(8, 3, 512, 512)
    outs = model(x)
    print(outs.size())


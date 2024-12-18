import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# from torchinfo import summary

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
        self.resnet4_1 = Resnet2(in_channel=64, out_channel=128)
        self.resnet4_2 = Resnet1(in_channel=128, out_channel=128)
        self.resnet4_3 = Resnet1(in_channel=128, out_channel=128)
        self.resnet5_1 = Resnet2(in_channel=128, out_channel=256)
        self.resnet5_2 = Resnet1(in_channel=256, out_channel=256)
        self.resnet5_3 = Resnet1(in_channel=256, out_channel=256)
        self.layer1.apply(weights_init)

    def forward(self, x):
        outs = []
        out = self.layer1(x)
        out = self.resnet1_1(out)
        out = self.resnet1_2(out)
        out = self.resnet1_3(out)
        # print("-------")
        # print(out.size())
        outs.append(out)
        out = self.resnet2_1(out)
        out = self.resnet2_2(out)
        out = self.resnet2_3(out)
        # print(out.size())
        outs.append(out)
        out = self.resnet3_1(out)
        out = self.resnet3_2(out)
        out = self.resnet3_3(out)
        # print(out.size())
        outs.append(out)
        out = self.resnet4_1(out)
        out = self.resnet4_2(out)
        out = self.resnet4_3(out)
        # print(out.size())
        outs.append(out)
        out = self.resnet5_1(out)
        out = self.resnet5_2(out)
        out = self.resnet5_3(out)
        # print(out.size())
        # print("-------")
        outs.append(out)
        return outs


class LCL(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LCL, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1, dilation=1),
            #nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.layer1.apply(weights_init)
    def forward(self, x):
        out = self.layer1(x)
        # print("-----")
        # print(out.size())
        # print("-----")
        return out


class Sbam(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Sbam, self).__init__()
        self.hl_layer = nn.Sequential(
          nn.UpsamplingBilinear2d(scale_factor=2),
          nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1),
          nn.BatchNorm2d(out_channel),
          nn.ReLU(inplace=True)
        )
        self.ll_layer = nn.Sequential(
          nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1),
          nn.BatchNorm2d(out_channel),
          nn.Sigmoid()  # ll = torch.sigmoid(ll)
        )
        self.hl_layer.apply(weights_init)
        self.ll_layer.apply(weights_init)
    def forward(self, hl,ll):
        hl = self.hl_layer(hl)
        # print(hl.size())
        ll_1 = ll
        ll = self.ll_layer(ll)
        # print(ll.size())
        hl_1 = hl*ll
        out = ll_1+hl_1
        return out

class ALCL_No_Sigmoid(nn.Module):
    def __init__(self):
        super(ALCL_No_Sigmoid, self).__init__()
        self.stage = Stage()
        self.lcl5 = LCL(256, 256)
        self.lcl4 = LCL(128, 128)
        self.lcl3 = LCL(64, 64)
        self.lcl2 = LCL(32, 32)
        self.lcl1 = LCL(16, 16)
        self.sbam4 = Sbam(256, 128)
        self.sbam3 = Sbam(128, 64)
        self.sbam2 = Sbam(64, 32)
        self.sbam1 = Sbam(32, 16)

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            # nn.Sigmoid()
        )
        self.layer.apply(weights_init)

    def forward(self, x):
        outs = self.stage(x)
        out5 = self.lcl5(outs[4])
        # print(out5.size())
        out4 = self.lcl4(outs[3])
        # print(out4.size())
        out3 = self.lcl3(outs[2])
        # print(out3.size())
        out2 = self.lcl2(outs[1])
        # print(out2.size())
        out1 = self.lcl1(outs[0])
        # print(out1.size())
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
    return

if __name__ == '__main__':
    model = ALCL_No_Sigmoid()
    x = torch.rand(8, 3, 512, 512)
    outs = model(x)
    print(outs.size())

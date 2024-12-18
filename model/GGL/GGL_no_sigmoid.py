import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad

class SEAttention(nn.Module):

    def __init__(self, channel,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8): 
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

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


# layer2_1 #layer3_1#layer4_1#layer5_1
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


class Resnet3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.SEAttention = SEAttention(channel=out_channel, reduction=4)
        self.relu = nn.ReLU(inplace=True)
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.SEAttention(out)
        out = self.layer2(out)
        out += identity
        return self.relu(out)

class Res(nn.Module):
    def __init__(self, befor_channel,after_channel):
        super(Res, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=befor_channel, out_channels=after_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(after_channel),
            nn.ReLU(inplace=True),
            SEAttention(channel=after_channel, reduction=4),
            nn.Conv2d(in_channels=after_channel, out_channels=after_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(after_channel),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=2*after_channel, out_channels=after_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(after_channel),
            nn.ReLU(inplace=True),
            SEAttention(channel=after_channel, reduction=4),
            nn.Conv2d(in_channels=after_channel, out_channels=after_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(after_channel),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x, x1):
        x1 = self.layer1(x1)
        # print(x1.size())
        con = torch.cat([x, x1], 1)
        identity = x
        out = self.layer2(con)
        out = out + identity
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
        self.resnet1_2 = Resnet3(in_channel=16, out_channel=16)
        self.resnet1_3 = Resnet3(in_channel=16, out_channel=16)
        self.Res1 = Res(befor_channel=3, after_channel=16)
        self.resnet2_1 = Resnet2(in_channel=16, out_channel=32)
        self.resnet2_2 = Resnet3(in_channel=32, out_channel=32)
        self.resnet2_3 = Resnet3(in_channel=32, out_channel=32)
        self.Res2 = Res(befor_channel=3, after_channel=32)
        self.resnet3_1 = Resnet2(in_channel=32, out_channel=64)
        self.resnet3_2 = Resnet3(in_channel=64, out_channel=64)
        self.resnet3_3 = Resnet3(in_channel=64, out_channel=64)
        self.Res3 = Res(befor_channel=3, after_channel=64)
        self.resnet4_1 = Resnet2(in_channel=64, out_channel=128)
        self.resnet4_2 = Resnet3(in_channel=128, out_channel=128)
        self.resnet4_3 = Resnet3(in_channel=128, out_channel=128)
        self.Res4 = Res(befor_channel=3, after_channel=128)
        self.resnet5_1 = Resnet2(in_channel=128, out_channel=256)
        self.resnet5_2 = Resnet3(in_channel=256, out_channel=256)
        self.resnet5_3 = Resnet3(in_channel=256, out_channel=256)
        self.Res5 = Res(befor_channel=3, after_channel=256)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer1.apply(weights_init)

    def forward(self, x):
        x_g = gradient_1order(x)
        outs = []
        out = self.layer1(x)
        # print(out_1.size())
        out = self.resnet1_1(out)
        out = self.resnet1_2(out)
        out = self.resnet1_3(out)
        # print(out.size())
        #out_1 = self.pool(out_1)
        out = self.Res1(out, x_g)
        # print("-------")
        # print(out.size())
        outs.append(out)
        out = self.resnet2_1(out)
        out = self.resnet2_2(out)
        out = self.resnet2_3(out)
        x1 = self.pool(x_g)
        out = self.Res2(out, x1)
        # print(out.size())
        outs.append(out)
        out = self.resnet3_1(out)
        out = self.resnet3_2(out)
        out = self.resnet3_3(out)
        # print(out.size())
        x2 = self.pool(x1)
        # print(x2.size())
        out = self.Res3(out, x2)
        # print(out.size())
        outs.append(out)
        out = self.resnet4_1(out)
        out = self.resnet4_2(out)
        out = self.resnet4_3(out)
        # print(out.size())
        x3 = self.pool(x2)
        # print(x3.size())
        out = self.Res4(out, x3)
        # print(out.size())
        outs.append(out)
        out = self.resnet5_1(out)
        out = self.resnet5_2(out)
        out = self.resnet5_3(out)
        x4 = self.pool(x3)
        out = self.Res5(out, x4)
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
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, padding=1, stride=1, dilation=1),
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
        self.hl_layer_2 = ChannelAttention(out_channel)
        self.ll_layer = SpatialAttention()
        # self.ll_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.Sigmoid()  # ll = torch.sigmoid(ll)
        # )
        self.hl_layer.apply(weights_init)
    def forward(self, hl,ll):
        hl = self.hl_layer(hl)
        # print(hl.size())
        ll_1 =ll * self.hl_layer_2(hl)

        ll = self.ll_layer(ll)
        # print(ll.size())
        hl_1 = hl * ll
        out = ll_1 + hl_1
        return out

class GGL_No_Sigmoid(nn.Module):
    def __init__(self):
        super(GGL_No_Sigmoid, self).__init__()
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
    # elif isinstance(m, nn.Linear):
    #     nn.init.xavier_uniform_(m.weight)
    #     nn.init.constant_(m.bias, 0)
    return


# if __name__ == '__main__':
#     model = GGL_No_Sigmoid()
#     x = torch.rand(8, 3, 512, 512)
#     x_g = gradient_1order(x)
#     outs = model(x,x_g)
#     print(outs.size())

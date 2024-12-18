import torch
import torch.nn as nn
from mm.attention.SEAttention import SEAttention
import pywt
from torch.nn import init
from torch.autograd import Function


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


################################################################################################
class LH_DWT_2D_attation(nn.Module):
    def __init__(self, wave):
        super(LH_DWT_2D_attation, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])  
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))  
        self.w_lh = self.w_lh.to(dtype=torch.float32)

    def forward(self, x):
        return LH_DWT_Function_attation.apply(x, self.w_lh)


class LH_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_lh):  
        x = x.contiguous()
        ctx.save_for_backward(w_lh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = torch.nn.functional.pad(x, (1, 0, 1, 0))  
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=1, padding=0, groups=dim)
        x = x_lh
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_lh = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)  
            filters = w_lh  # torch.Size([3, 1, 2, 2])
            filters = filters.repeat(C, 1, 1, 1).to(dtype=torch.float16)  
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2)  
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)  
        return dx, None


########################################################################################
class HL_DWT_2D_attation(nn.Module):
    def __init__(self, wave):
        super(HL_DWT_2D_attation, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])  

        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  

        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))  
        self.w_hl = self.w_hl.to(dtype=torch.float32)

    def forward(self, x):
        return HL_DWT_Function_attation.apply(x, self.w_hl)


class HL_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_hl):  
        x = x.contiguous()
        ctx.save_for_backward(w_hl)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = torch.nn.functional.pad(x, (1, 0, 1, 0))  
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=1, padding=0, groups=dim)
        x = x_hl
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_hl = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)  
            filters = w_hl
            filters = filters.repeat(C, 1, 1, 1).to(dtype=torch.float16) 
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2)  
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)  
        return dx, None


#######################################################################################
class HH_DWT_2D_attation(nn.Module):
    def __init__(self, wave):
        super(HH_DWT_2D_attation, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])  
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))  
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return HH_DWT_Function_attation.apply(x, self.w_hh)


class HH_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_hh):  
        x = x.contiguous()
        ctx.save_for_backward(w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = torch.nn.functional.pad(x, (1, 0, 1, 0))  
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=1, padding=0,
                                          groups=dim)  
        x = x_hh
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_hh = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)  
            filters = w_hh  # torch.Size([3, 1, 2, 2])
            filters = filters.repeat(C, 1, 1, 1).to(dtype=torch.float16)  
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2)  
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)  
        return dx, None


#######################################################################################
class LL_DWT_2D_attation(nn.Module):
    def __init__(self, wave):
        super(LL_DWT_2D_attation, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])  
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))  
        self.w_ll = self.w_ll.to(dtype=torch.float32)

    def forward(self, x):
        return LL_DWT_Function_attation.apply(x, self.w_ll)


class LL_DWT_Function_attation(Function):
    @staticmethod
    def forward(ctx, x, w_ll):  
        x = x.contiguous()
        ctx.save_for_backward(w_ll)
        ctx.shape = x.shape
        dim = x.shape[1]
        x = torch.nn.functional.pad(x, (1, 0, 1, 0))  
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=1, padding=0,
                                          groups=dim)  
        x = x_ll
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            dx = dx.view(B, 1, -1, H, W)
            dx = dx.transpose(1, 2).reshape(B, -1, H, W)  
            filters = w_ll  # torch.Size([3, 1, 2, 2])
            filters = filters.repeat(C, 1, 1, 1).to(dtype=torch.float16)  
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=1, groups=C)
            dx = torch.cat((dx[:, :, :0], dx[:, :, 1:]), dim=2) 
            dx = torch.cat((dx[:, :, :, :0], dx[:, :, :, 1:]), dim=3)  
        return dx, None


#########################################################################################

class FrequencyAttention(nn.Module):  #

    def __init__(self, in_channel):
        super().__init__()
        # self.HH_reduce = nn.Sequential(
        #     nn.Conv2d(in_channel, 1, kernel_size=3, padding=1, stride=1),
        #     nn.Sigmoid(),
        # )
        self.HH_reduce = SpatialAttention()
        # self.LH_reduce = nn.Sequential(
        #     nn.Conv2d(in_channel, 1, kernel_size=3, padding=1, stride=1),
        #     nn.Sigmoid(),
        # )
        self.LH_reduce = SpatialAttention()
        # self.HL_reduce = nn.Sequential(
        #     nn.Conv2d(in_channel, 1, kernel_size=3, padding=1, stride=1),
        #     nn.Sigmoid(),
        # )
        self.HL_reduce = SpatialAttention()
        # self.LL_reduce = nn.Sequential(
        #     nn.Conv2d(in_channel, 1, kernel_size=3, padding=1, stride=1),
        #     nn.Sigmoid(),
        # )
        self.LL_reduce = SpatialAttention()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channel),
        )
        self.relu = nn.ReLU(inplace=False)
        # self.fc =nn.Linear(in_channel*3, in_channel, bias=False)
        self.LH_DWT_2D_attation = LH_DWT_2D_attation('haar')
        self.HL_DWT_2D_attation = HL_DWT_2D_attation('haar')
        self.HH_DWT_2D_attation = HH_DWT_2D_attation('haar')
        self.LL_DWT_2D_attation = LL_DWT_2D_attation('haar')
        self.init_weights()

    def forward(self, x):
        out_LH = self.LH_DWT_2D_attation(x)
        out_LH = self.LH_reduce(out_LH)
        x_LH = x * out_LH
        # print(x_LH.size())
        out_HL = self.HL_DWT_2D_attation(x)
        out_HL = self.HL_reduce(out_HL)
        x_HL = x * out_HL
        out_HH = self.HH_DWT_2D_attation(x)
        out_HH = self.HH_reduce(out_HH)
        x_HH = x * out_HH
        out_LL = self.LL_DWT_2D_attation(x)
        out_LL = self.LL_reduce(out_LL)
        x_LL = x * out_LL
        x_out = x_LH + x_HL + x_HH + x_LL
        x_out = self.relu(x + self.conv_res(x_out))
        return x_out

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


############################################################################################

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])  

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))  
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0)) 
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))  
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))  

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):  
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]

        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, padding=0, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, padding=0,
                                          groups=dim)  
        x = torch.cat([x_lh, x_hl, x_hh], dim=1)
        # x =  x_hh
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape

            dx = dx.view(B, 3, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_lh, w_hl, w_hh], dim=0)
            filters = filters.repeat(C, 1, 1, 1).to(dtype=torch.float16)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class Hfrequencyfeature(nn.Module):  #

    def __init__(self):
        super().__init__()

        self.DWT_2D = DWT_2D('haar')

    def forward(self, x):  # x:torch.Size([8, 16, 512, 512])
        out = self.DWT_2D(x)  # torch.Size([8, 48, 512, 512])
        return out


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):  
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    return


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##############################################################################
class RFB_modified(nn.Module): 
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(5, 5), padding=2),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.FrequencyAttention = FrequencyAttention(in_channel=out_channel)
        self.SEAttention = SEAttention(channel=out_channel, reduction=4)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x_cat = self.FrequencyAttention(x_cat)
        # print(x_cat.shape)
        x_cat = self.SEAttention(x_cat)
        x = self.relu(x+ self.conv_res(x_cat))
        return x


class RFB_modified_LCL(nn.Module):  
    def __init__(self, in_channel, out_channel):
        super(RFB_modified_LCL, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=(5, 5), padding=2),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        # print(x_cat.size())
        x_cat = self.relu(x_cat)
        return x_cat


#############################################################################################################
class Resnet1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resnet1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace=False)
       
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
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channel)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )
        self.relu = nn.ReLU(inplace=False)
       
        self.layer1.apply(weights_init)
        self.layer2.apply(weights_init)

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        identity = self.layer2(identity)
        # print("out:",out.size())
        # print("identity:",identity.size())
        out += identity
        return self.relu(out)


class Hfrequency(nn.Module):
    def __init__(self):
        super(Hfrequency, self).__init__()
        self.Hfrequencyfeature = Hfrequencyfeature()

    def forward(self, x, out_2):
        out_1 = self.Hfrequencyfeature(x)
        out = torch.cat([out_1, out_2], dim=1)
        return out


class Stage(nn.Module):
    def __init__(self):
        super(Stage, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False)
        )
        self.resnet1_1 = Resnet1(in_channel=16, out_channel=16)
        self.resnet1_2 = RFB_modified(in_channel=16, out_channel=16)
        self.resnet1_3 = RFB_modified(in_channel=16, out_channel=16)
        self.layer1_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )


        self.resnet2_1 = Resnet2(in_channel=16, out_channel=23)
        self.hfrequency = Hfrequency()
        self.resnet2_2 = RFB_modified(in_channel=32, out_channel=32)
        self.resnet2_3 = RFB_modified(in_channel=32, out_channel=32)
        self.layer2_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )


        self.resnet3_1 = Resnet2(in_channel=32, out_channel=64)
        self.resnet3_2 = RFB_modified(in_channel=64, out_channel=64)
        self.resnet3_3 = RFB_modified(in_channel=64, out_channel=64)
        self.layer3_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )


        self.resnet4_1 = Resnet2(in_channel=64, out_channel=64)
        self.resnet4_2 = RFB_modified(in_channel=64, out_channel=64)
        self.resnet4_3 = RFB_modified(in_channel=64, out_channel=64)
        self.layer4_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
        )


        self.resnet5_1 = Resnet2(in_channel=64, out_channel=64)
        self.resnet5_2 = RFB_modified(in_channel=64, out_channel=64)
        self.resnet5_3 = RFB_modified(in_channel=64, out_channel=64)
        self.layer5_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=False),

        )


        
        self.layer1.apply(weights_init)
        self.layer1_4.apply(weights_init)
        self.layer2_4.apply(weights_init)
        self.layer3_4.apply(weights_init)
        self.layer4_4.apply(weights_init)
        self.layer5_4.apply(weights_init)

    def forward(self, x):
        outs = []
        out = self.layer1(x)

        # print(out_1.size())
        out = self.resnet1_1(out)
        out = self.resnet1_2(out)
        out = self.resnet1_3(out)
        out_1 = self.layer1_4(out)

        outs.append(out)
        out = self.resnet2_1(out)
        out = self.hfrequency(x,out)
        out = self.resnet2_2(out)
        out = self.resnet2_3(out)
        out_2 = self.layer2_4(out)

        outs.append(out)
        out = self.resnet3_1(out)
        out = self.resnet3_2(out)
        out = self.resnet3_3(out)
        out_3 = self.layer3_4(out)

        outs.append(out)
        out = self.resnet4_1(out)
        out = self.resnet4_2(out)
        out = self.resnet4_3(out)
        out_4 = self.layer4_4(out)

        outs.append(out)
        out = self.resnet5_1(out)
        out = self.resnet5_2(out)
        out = self.resnet5_3(out)
        out = torch.cat([out, out_4, out_3,out_2,out_1], dim=1)
        out = self.layer5_4(out)

        outs.append(out)
        return outs

class Sbam(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Sbam, self).__init__()
        self.hl_up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.hl_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )
        self.concat_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel+out_channel, out_channels=in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=False),
        )

       
        self.hl_layer.apply(weights_init)
        self.concat_layer.apply(weights_init)

    def forward(self, hl, ll):
        hl = self.hl_up(hl)
        concat = torch.cat((hl, ll), 1)
        k = self.concat_layer(concat)
        hl = hl + k
        hl = self.hl_layer(hl)
        out = ll + hl
        return out


class MSDA_No_Sigmoid(nn.Module):
    def __init__(self):
        super(MSDA_No_Sigmoid, self).__init__()
        self.stage = Stage()
        self.mlcl5 = RFB_modified_LCL(64, 64)
        self.mlcl4 = RFB_modified_LCL(64, 64)
        self.mlcl3 = RFB_modified_LCL(64, 64)
        self.mlcl2 = RFB_modified_LCL(32, 32)
        self.mlcl1 = RFB_modified_LCL(16, 16)

        self.sbam4 = Sbam(64, 64)
        self.sbam3 = Sbam(64, 64)
        self.sbam2 = Sbam(64, 32)
        self.sbam1 = Sbam(32, 16)

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            # nn.Sigmoid()
        )
        
        self.layer.apply(weights_init)

    def forward(self, x):
        outs = self.stage(x)

        out5 = self.mlcl5(outs[4])
        out4 = self.mlcl4(outs[3])
        out3 = self.mlcl3(outs[2])
        out2 = self.mlcl2(outs[1])
        out1 = self.mlcl1(outs[0])

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


if __name__ == '__main__':
    model = MSDA_No_Sigmoid()
    x = torch.rand(8, 3, 512, 512)
    outs = model(x)
    print(outs.size())

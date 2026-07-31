import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# from utils.pooling import *
# from models.fusion import *
from einops import rearrange
import numbers
import copy
import pywt
import pywt.data


# -------wavelet--------
# def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
#     w = pywt.Wavelet(wave)
#     dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
#     dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
#     dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
#                                dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)
#
#     dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)
#
#     rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
#     rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
#     rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
#                                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)
#
#     rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)
#
#     return dec_filters, rec_filters
#
#
# def wavelet_transform(x, filters):
#     b, c, h, w = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
#     x = x.reshape(b, c, 4, h // 2, w // 2)
#     return x
#
#
# def inverse_wavelet_transform(x, filters):
#     b, c, _, _ = x.shape
#     pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
#     # x = x.reshape(b, c * 4, h_half, w_half)
#     x = F.conv_transpose2d(x, filters, stride=2, groups=c // 4, padding=pad)
#     return x


class wavelet(nn.Module):
    def __init__(self, wave='haar', channels=1, dtype=torch.float32):
        """
        可作为 nn.Module 使用，也兼容旧版函数式调用。
        wave: 小波类型，例如 'haar', 'db1', 'db2'
        channels: 输入通道数
        """
        super().__init__()

        self.wave = wave
        self.channels = channels

        dec_filters, rec_filters = self.create_wavelet_filter(
            wave=wave,
            channels=channels,
            dtype=dtype
        )

        self.register_buffer("dec_filters", dec_filters)
        self.register_buffer("rec_filters", rec_filters)

    @staticmethod
    def create_wavelet_filter(
            wave,
            in_size=None,
            out_size=None,
            type=torch.float32,
            channels=None,
            dtype=None
    ):
        """
        兼容两种调用方式：

        旧方式：
            wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)

        新方式：
            wavelet.create_wavelet_filter(wave='db1', channels=in_channels, dtype=torch.float32)
        """

        # 兼容新写法 channels=xxx
        if channels is not None:
            in_size = channels

        if in_size is None:
            raise ValueError("必须指定 in_size 或 channels")

        # 如果没有指定 out_size，默认和 in_size 一致
        if out_size is None:
            out_size = in_size

        # 兼容 dtype=xxx
        if dtype is not None:
            type = dtype

        w = pywt.Wavelet(wave)

        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)

        dec_filters = torch.stack([
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ], dim=0)

        dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

        rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
        rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])

        rec_filters = torch.stack([
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ], dim=0)

        rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

        return dec_filters, rec_filters

    @staticmethod
    def wavelet_transform(x, filters):
        """
        兼容旧函数 wavelet_transform(x, filters)

        输入:
            x: [B, C, H, W]

        输出:
            x: [B, C, 4, H/2, W/2]
        """
        b, c, h, w = x.shape

        pad = (
            filters.shape[2] // 2 - 1,
            filters.shape[3] // 2 - 1
        )

        x = F.conv2d(
            x,
            filters,
            stride=2,
            groups=c,
            padding=pad
        )

        x = x.reshape(b, c, 4, h // 2, w // 2)

        return x

    @staticmethod
    def inverse_wavelet_transform(x, filters):
        """
        兼容旧函数 inverse_wavelet_transform(x, filters)

        输入:
            x: [B, 4*C, H/2, W/2]

        输出:
            x: [B, C, H, W]
        """
        b, c, h, w = x.shape

        pad = (
            filters.shape[2] // 2 - 1,
            filters.shape[3] // 2 - 1
        )

        x = F.conv_transpose2d(
            x,
            filters,
            stride=2,
            groups=c // 4,
            padding=pad
        )

        return x

    def forward(self, x):
        """
        nn.Module 方式调用：
            wt = wavelet(wave='db1', channels=C)
            y = wt(x)
        """
        return self.wavelet_transform(x, self.dec_filters)

    def inverse(self, x):
        """
        nn.Module 方式逆变换：
            x_rec = wt.inverse(y)

        支持:
            y: [B, C, 4, H/2, W/2]
            或 [B, 4*C, H/2, W/2]
        """
        if x.dim() == 5:
            b, c, four, h_half, w_half = x.shape
            assert four == 4
            x = x.reshape(b, c * 4, h_half, w_half)

        return self.inverse_wavelet_transform(x, self.rec_filters)


# -----pooling-----------
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale, requires_grad=True)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PWD2d(nn.Module):  # 小波+卷积下采样核
    def __init__(self, in_channels, out_channels, wt_type='db1'):
        super(PWD2d, self).__init__()

        assert in_channels * 4 == out_channels

        self.in_channels = in_channels
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)

        self.para_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=2, stride=2, groups=in_channels),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU())  # 分组2*2卷积
        self.ca = ChannelAttention(out_channels)

        self.wavelet_scale = _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
        self.conv1x1 = nn.Conv2d(out_channels, in_channels * 2, kernel_size=1, stride=1)
        self.bnorm = nn.BatchNorm2d(in_channels * 2)
        self.act = nn.ReLU()

    def forward(self, x):
        curr_x_ll = x

        x = self.para_conv(x)
        ca = self.ca(x)
        x = x * ca

        curr_shape = curr_x_ll.shape
        if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
            curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
            curr_x_ll = F.pad(curr_x_ll, curr_pads)

        curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
        shape_x = curr_x.shape
        curr_x_wt = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])

        curr_x_wt = curr_x_wt * ca
        curr_x_wt = self.wavelet_scale(curr_x_wt)

        out = x + curr_x_wt
        out = self.act(self.bnorm(self.conv1x1(out)))

        return out


# --------fusion---------#
class LowPassConvGenerator(nn.Module):
    def __init__(self, in_channel, num_k=2, ratio=8):
        super(LowPassConvGenerator, self).__init__()
        self.num_k = num_k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1_list = nn.ModuleList(
            [nn.Conv2d(in_channel // num_k * 2, in_channel // ratio, 1, bias=False) for _ in range(num_k)])
        self.fc2_list = nn.ModuleList(
            [nn.Conv2d(in_channel // ratio, 9 * in_channel, 1, bias=False) for _ in range(num_k)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        kernels = []

        for i in range(self.num_k):
            start_idx = i * (c // self.num_k)
            end_idx = (i + 1) * (c // self.num_k)
            x_i = x[:, start_idx:end_idx, :, :]

            avg_feature_x = self.avg_pool(x_i)  # b, c/num_k, 1, 1
            max_feature_x = self.max_pool(x_i)  # b, c/num_k, 1, 1

            fc1 = self.fc1_list[i]
            fc2 = self.fc2_list[i]

            combined_feature = torch.cat([avg_feature_x, max_feature_x], dim=1)
            kernel = fc2(self.relu(fc1(combined_feature)))  # (b, 9c, 1, 1)

            kernel = kernel.reshape(b, c, 9, 1)
            kernel = F.softmax(kernel, dim=2)
            kernel = kernel.reshape(b, c, 3, 3)[0].unsqueeze(1)  # (c/num_k, 1, 3, 3)

            kernels.append(kernel)
        final_kernel = torch.cat(kernels, dim=0)  # (num_k * c, 1, 3, 3)

        return final_kernel


class HighPassConvGenerator(nn.Module):
    def __init__(self, in_channel, num_k=2, kernel_size=3, ratio=8):
        super(HighPassConvGenerator, self).__init__()
        self.num_k = num_k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1_list = nn.ModuleList(
            [nn.Conv2d(in_channel // num_k * 2, in_channel // ratio, 1, bias=False) for _ in range(num_k)])
        self.fc2_list = nn.ModuleList(
            [nn.Conv2d(in_channel // ratio, 9 * in_channel, 1, bias=False) for _ in range(num_k)])
        self.relu = nn.ReLU(inplace=True)

        self.identity_kernel = nn.Parameter(torch.tensor([[0, 0, 0],
                                                          [0, 1, 0],
                                                          [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                            requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape  # 64,64,64
        kernels = []

        for i in range(self.num_k):
            start_idx = i * (c // self.num_k)
            end_idx = (i + 1) * (c // self.num_k)
            x_i = x[:, start_idx:end_idx, :, :]

            avg_feature_x = self.avg_pool(x_i)  # b, c/num_k, 1, 1
            max_feature_x = self.max_pool(x_i)  # b, c/num_k, 1, 1
            fc1 = self.fc1_list[i]
            fc2 = self.fc2_list[i]

            combined_feature = torch.cat([avg_feature_x, max_feature_x], dim=1)
            kernel = fc2(self.relu(fc1(combined_feature)))  # (b, 9c, 1, 1)

            kernel = kernel.reshape(b, c, 9, 1)
            kernel = F.softmax(kernel, dim=2)
            kernel = kernel.reshape(b, c, 3, 3)[0].unsqueeze(1)  # (c/num_k, 1, 3, 3)

            kernels.append(self.identity_kernel - kernel)
        final_kernel = torch.cat(kernels, dim=0)  # (num_k * c, 1, 3, 3)

        return final_kernel


class Freq_Fusion(nn.Module):
    def __init__(self, c_high, c_low, c_out, num_k=2):
        super(Freq_Fusion, self).__init__()
        assert c_high == c_low * 2, 'c_high must be 2 * c_low'
        self.num_k = num_k
        self.c_high = c_high
        self.c_low = c_low
        self.c_out = c_out

        self.init_high_conv = nn.Conv2d(c_high, c_high // 2, 1, 1, padding='same')
        self.y_conv1 = nn.Conv2d(c_high, c_low, 1, 1)
        self.x_conv2 = nn.Conv2d(c_high, c_low, 1, 1)
        self.HPG_conv = nn.Sequential(nn.Conv2d(c_low, c_low, 7, 1, padding='same', groups=c_low),
                                      nn.BatchNorm2d(c_low),
                                      nn.ReLU(inplace=True))
        self.LPG_conv = nn.Sequential(nn.Conv2d(c_low, c_low, 3, 1, padding='same', groups=c_low),
                                      nn.BatchNorm2d(c_low),
                                      nn.ReLU(inplace=True))
        self.Cat_conv = nn.Sequential(
            nn.Conv2d(c_high + c_low, c_high + c_low, 3, 1, padding='same', groups=c_high + c_low),
            nn.BatchNorm2d(c_high + c_low),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_high + c_low, c_low, 1, 1, padding='same'),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True)
        )
        self.res_conv = nn.Conv2d(c_low, c_low * 2, 1, 1, padding=0)
        self.HPG = HighPassConvGenerator(c_low, num_k=self.num_k)  # (c_low, 1, 3, 3)
        self.LPG = LowPassConvGenerator(c_low, num_k=self.num_k)  # (c_low, 1, 3, 3)

        self.x_conv_fhnorm = nn.BatchNorm2d(self.c_low * 2)
        self.y_conv_flnorm = nn.BatchNorm2d(self.c_low * 2)

        self.plus_conv = nn.Sequential(
            nn.Conv2d(c_low, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_low, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_low, y_high):
        y_high_up = F.interpolate(y_high, scale_factor=2, mode='bilinear', align_corners=True)
        cat_fuse = torch.cat([x_low, y_high_up], dim=1)  # c_low+c_high,64,64
        cat_fuse = self.Cat_conv(cat_fuse)

        x_h = self.HPG_conv(x_low)
        x_l = self.LPG_conv(x_low)
        kernel_L = self.LPG(x_l)
        kernel_H = self.HPG(x_h)  # 64,1,3,3

        res = x_low
        x_low = F.conv2d(x_low, kernel_H, padding=1, groups=kernel_H.shape[0] // self.num_k)  # 64,64,64
        x_low = self.x_conv_fhnorm(x_low + self.res_conv(res))
        x_low = self.x_conv2(x_low)

        y_high_up = self.init_high_conv(y_high_up)
        y_high_up = F.conv2d(y_high_up, kernel_L, padding=1, groups=kernel_L.shape[0] // self.num_k)  # 64,64,64
        y_high_up = self.y_conv_flnorm(y_high_up)
        y_high_up = self.y_conv1(y_high_up)

        plus_fuse = x_low + y_high_up
        plus_fuse = self.plus_conv(plus_fuse)
        fuse = plus_fuse + cat_fuse
        fuse = self.out_conv(fuse)

        return fuse


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class GSFANet_No_Sigmoid(nn.Module):
    def __init__(self, size=256, input_channels=3, block=ResNet):
        super().__init__()
        param_channels = [16, 32, 64, 128]  # down3
        param_blocks = [2, 2, 2, 2]

        self.pool1 = PWD2d(param_channels[0], param_channels[0] * 4)
        self.pool2 = PWD2d(param_channels[1], param_channels[1] * 4)
        self.pool3 = PWD2d(param_channels[2], param_channels[2] * 4)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block, param_blocks[0])
        self.encoder_1 = self._make_layer(param_channels[0] * 2, param_channels[1], block, param_blocks[1])
        self.encoder_2 = self._make_layer(param_channels[1] * 2, param_channels[2], block, param_blocks[2])

        self.middle_layer = self._make_layer(param_channels[2] * 2, param_channels[3], block, param_blocks[3])

        self.mla = MLA(channel_num=[param_channels[0], param_channels[1], param_channels[2], param_channels[3]],
                       patchSize=[32, 16, 8, 4], layer_num=2, size=size)

        self.decoder_2 = Freq_Fusion(param_channels[3], param_channels[2], param_channels[2])
        self.decoder_1 = Freq_Fusion(param_channels[2], param_channels[1], param_channels[1])
        self.decoder_0 = Freq_Fusion(param_channels[1], param_channels[0], param_channels[0])

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.final = nn.Conv2d(3, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, tag=True):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool1(x_e0))
        x_e2 = self.encoder_2(self.pool2(x_e1))

        x_m = self.middle_layer(self.pool3(x_e2))

        x_e0, x_e1, x_e2, x_m, _ = self.mla(x_e0, x_e1, x_e2, x_m)

        x_d2 = self.decoder_2(x_e2, x_m)
        x_d1 = self.decoder_1(x_e1, x_d2)
        x_d0 = self.decoder_0(x_e0, x_d1)

        if tag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2)], dim=1))
            return [mask0, mask1, mask2], output

        else:
            output = self.output_0(x_d0)
            return [], output


class MLA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], patchSize=[32, 16, 8, 4], layer_num=1, size=256):
        super().__init__()
        patchSize = [size // 8, size // 16, size // 32, size // 64]  # patch32(256), 64(512)
        overlap = [p // 2 for p in patchSize]  # p/2
        self.embeddings_1 = Channel_Embeddings(patchSize[0], in_channels=channel_num[0], overlap=overlap[0])
        self.embeddings_2 = Channel_Embeddings(patchSize[1], in_channels=channel_num[1], overlap=overlap[1])
        self.embeddings_3 = Channel_Embeddings(patchSize[2], in_channels=channel_num[2], overlap=overlap[2])
        self.embeddings_4 = Channel_Embeddings(patchSize[3], in_channels=channel_num[3], overlap=overlap[3])

        self.layer = nn.ModuleList()
        for _ in range(layer_num):
            layer = MLA_Block(channel_num, mode='kernel')
            self.layer.append(copy.deepcopy(layer))

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1)
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1)
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1)
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1)

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, sa_weight = layer_block(emb1, emb2, emb3, emb4)

        org_size = [(en1.shape[2], en1.shape[3]), (en2.shape[2], en2.shape[3]), (en3.shape[2], en3.shape[3]),
                    (en4.shape[2], en4.shape[3])]
        emb1 = self.reconstruct_1(emb1, size=org_size[0]) if emb1 is not None else None
        emb2 = self.reconstruct_2(emb2, size=org_size[1]) if emb2 is not None else None
        emb3 = self.reconstruct_3(emb3, size=org_size[2]) if emb3 is not None else None
        emb4 = self.reconstruct_4(emb4, size=org_size[3]) if emb4 is not None else None

        out1 = emb1 + en1 if en1 is not None else None
        out2 = emb2 + en2 if en2 is not None else None
        out3 = emb3 + en3 if en3 is not None else None
        out4 = emb4 + en4 if en4 is not None else None

        return out1, out2, out3, out4, sa_weight


class MLA_Block(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], mode='kernel'):
        super().__init__()
        self.msa = MSA(channel_num)  # sep or sep_gate or same_conv # 修改这里
        self.mca = MCA(channel_num, mode=mode)
        self.mca_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.mca_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.mca_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.mca_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

    def forward(self, emb1, emb2, emb3, emb4):
        res1, res2, res3, res4 = emb1, emb2, emb3, emb4
        ca1, ca2, ca3, ca4 = self.mca(emb1, emb2, emb3, emb4)
        ca1 = ca1 + res1
        ca2 = ca2 + res2
        ca3 = ca3 + res3
        ca4 = ca4 + res4

        ca1 = self.mca_norm1(ca1) if ca1 is not None else None
        ca2 = self.mca_norm2(ca2) if ca2 is not None else None
        ca3 = self.mca_norm3(ca3) if ca3 is not None else None
        ca4 = self.mca_norm4(ca4) if ca4 is not None else None

        # return ca1, ca2, ca3, ca4
        sa1, sa2, sa3, sa4, sa_weight = self.msa(ca1, ca2, ca3, ca4)

        return sa1, sa2, sa3, sa4, sa_weight


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, size=None):
        if x is None:
            return None

        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Channel_Embeddings(nn.Module):
    def __init__(self, patchsize, in_channels, overlap):
        super().__init__()
        # patch_size = _pair(patchsize)
        # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patchsize,
                                          stride=patchsize - overlap)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        # self.dropout = nn.Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # b, c, 31, 31
        return x


class MSA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128]):  # sum[16, 32, 64, 128, 256]
        super().__init__()
        self.channel_size = channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]  # 240
        self.conv_emb = nn.Conv2d(self.channel_size, self.channel_size // 8, 1, 1, padding='same')
        self.emb_norm = nn.BatchNorm2d(self.channel_size // 8)
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(channel_num[3], channel_num[3] // 8, 1, 1, padding='same'),  # channel_num[3] // 8
            nn.BatchNorm2d(channel_num[3] // 8),
            nn.ReLU()
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(channel_num[2], channel_num[2], 3, 1, padding='same', groups=channel_num[2]),
            nn.Conv2d(channel_num[2], channel_num[2] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[2] // 8),
            nn.ReLU()
        )
        self.Conv_5 = nn.Sequential(
            nn.Conv2d(channel_num[1], channel_num[1], 5, 1, padding='same', groups=channel_num[1]),
            nn.Conv2d(channel_num[1], channel_num[1] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[1] // 8),
            nn.ReLU()
        )
        self.Conv_7 = nn.Sequential(
            nn.Conv2d(channel_num[0], channel_num[0], 7, 1, padding='same', groups=channel_num[0]),
            nn.Conv2d(channel_num[0], channel_num[0] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[0] // 8),
            nn.ReLU()
        )
        self.se = SE_Block(self.channel_size // 8 * 2, ratio=10)  # 5, 10
        self.final_Conv = nn.Conv2d(self.channel_size // 8 * 2, 1, 1, 1, padding='same')

        self.sig = nn.Sigmoid()

    def forward(self, emb1, emb2, emb3, emb4):
        emb = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        emb = self.conv_emb(emb)
        emb = self.emb_norm(emb)
        emb_1 = self.Conv_7(emb1)
        emb_3 = self.Conv_5(emb2)
        emb_5 = self.Conv_3(emb3)
        emb_7 = self.Conv_1(emb4)

        sa = self.final_Conv(self.se(torch.cat([emb_1, emb_3, emb_5, emb_7, emb], dim=1)))
        sa = self.sig(sa)

        emb_1 = sa * emb1
        emb_2 = sa * emb2
        emb_3 = sa * emb3
        emb_4 = sa * emb4

        emb_1 = emb_1 + emb1
        emb_2 = emb_2 + emb2
        emb_3 = emb_3 + emb3
        emb_4 = emb_4 + emb4

        return emb_1, emb_2, emb_3, emb_4, sa


class MCA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], mode='kernel'):
        super().__init__()
        self.mode = mode
        self.channel_size = channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(self.channel_size, LayerNorm_type='WithBias')

        if self.mode == 'kernel':
            self.q1_Conv = nn.Conv2d(channel_num[0], channel_num[0], 1, 1)
            self.k1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)
            self.v1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)

            self.q2_Conv = nn.Conv2d(channel_num[1], channel_num[1], 1, 1)
            self.k2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)
            self.v2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)

            self.q3_Conv = nn.Conv2d(channel_num[2], channel_num[2], 1, 1)
            self.k3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)
            self.v3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)

            self.q4_Conv = nn.Conv2d(channel_num[3], channel_num[3], 1, 1)
            self.k4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)
            self.v4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)

            self.out_Conv1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
            self.out_Conv2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
            self.out_Conv3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
            self.out_Conv4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

            self.softmax = nn.Softmax(dim=1)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)

    def kernel_similarity(self, Gq, Gk, sigma=None):
        assert Gq.shape == Gk.shape

        diff = Gq - Gk
        if sigma is None:
            sigma = torch.std(diff, dim=2).unsqueeze(-1)
        distance = torch.norm(diff, p=2, dim=2).unsqueeze(-1)
        similarity = torch.exp(-distance ** 2 / (2 * sigma ** 2))  # GUSSIAN KERNEL

        return similarity

    def forward(self, emb1, emb2, emb3, emb4):
        b, c, h, w = emb1.shape
        emb_all = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        emb1_norm = self.attn_norm1(emb1) if emb1 is not None else None
        emb2_norm = self.attn_norm2(emb2) if emb2 is not None else None
        emb3_norm = self.attn_norm3(emb3) if emb3 is not None else None
        emb4_norm = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)

        if self.mode == 'kernel':
            emb1_gq = self.q1_Conv(emb1_norm)
            emb_all_gk1 = self.k1_Conv(emb_all)
            emb_all_gv1 = self.v1_Conv(emb_all)
            emb1_gq = rearrange(emb1_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk1 = rearrange(emb_all_gk1, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv1 = rearrange(emb_all_gv1, 'b c h w -> b c (h w)', h=h, w=w)
            emb1_gq, emb_all_gk1 = (F.normalize(emb1_gq, dim=-1), F.normalize(emb_all_gk1, dim=-1))

            emb2_gq = self.q2_Conv(emb2_norm)
            emb_all_gk2 = self.k2_Conv(emb_all)
            emb_all_gv2 = self.v2_Conv(emb_all)
            emb2_gq = rearrange(emb2_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk2 = rearrange(emb_all_gk2, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv2 = rearrange(emb_all_gv2, 'b c h w -> b c (h w)', h=h, w=w)
            emb2_gq, emb_all_gk2 = (F.normalize(emb2_gq, dim=-1), F.normalize(emb_all_gk2, dim=-1))

            emb3_gq = self.q3_Conv(emb3_norm)
            emb_all_gk3 = self.k3_Conv(emb_all)
            emb_all_gv3 = self.v3_Conv(emb_all)
            emb3_gq = rearrange(emb3_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk3 = rearrange(emb_all_gk3, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv3 = rearrange(emb_all_gv3, 'b c h w -> b c (h w)', h=h, w=w)
            emb3_gq, emb_all_gk3 = (F.normalize(emb3_gq, dim=-1), F.normalize(emb_all_gk3, dim=-1))

            emb4_gq = self.q4_Conv(emb4_norm)
            emb_all_gk4 = self.k4_Conv(emb_all)
            emb_all_gv4 = self.v4_Conv(emb_all)
            emb4_gq = rearrange(emb4_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk4 = rearrange(emb_all_gk4, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv4 = rearrange(emb_all_gv4, 'b c h w -> b c (h w)', h=h, w=w)
            emb4_gq, emb_all_gk4 = (F.normalize(emb4_gq, dim=-1), F.normalize(emb_all_gk4, dim=-1))

            sim1 = self.kernel_similarity(emb1_gq, emb_all_gk1)
            sim2 = self.kernel_similarity(emb2_gq, emb_all_gk2)
            sim3 = self.kernel_similarity(emb3_gq, emb_all_gk3)
            sim4 = self.kernel_similarity(emb4_gq, emb_all_gk4)

            sim1, sim2, sim3, sim4 = self.softmax(sim1), self.softmax(sim2), self.softmax(sim3), self.softmax(sim4)

            out1 = sim1 * emb_all_gv1
            out2 = sim2 * emb_all_gv2
            out3 = sim3 * emb_all_gv3
            out4 = sim4 * emb_all_gv4

            out1 = rearrange(out1, 'b c (h w) -> b c h w', h=h, w=w)
            out2 = rearrange(out2, 'b c (h w) -> b c h w', h=h, w=w)
            out3 = rearrange(out3, 'b c (h w) -> b c h w', h=h, w=w)
            out4 = rearrange(out4, 'b c (h w) -> b c h w', h=h, w=w)

            att_out1 = self.out_Conv1(out1)
            att_out2 = self.out_Conv2(out2)
            att_out3 = self.out_Conv3(out3)
            att_out4 = self.out_Conv4(out4)

            att_out1 = att_out1 + emb1 if emb1 is not None else None
            att_out2 = att_out2 + emb2 if emb2 is not None else None
            att_out3 = att_out3 + emb3 if emb3 is not None else None
            att_out4 = att_out4 + emb4 if emb4 is not None else None

            res1 = att_out1
            res2 = att_out2
            res3 = att_out3
            res4 = att_out4
            x1 = self.ffn_norm1(res1) if emb1 is not None else None
            x2 = self.ffn_norm2(res2) if emb2 is not None else None
            x3 = self.ffn_norm3(res3) if emb3 is not None else None
            x4 = self.ffn_norm4(res4) if emb4 is not None else None
            x1 = self.ffn1(x1) if emb1 is not None else None
            x2 = self.ffn2(x2) if emb2 is not None else None
            x3 = self.ffn3(x3) if emb3 is not None else None
            x4 = self.ffn4(x4) if emb4 is not None else None
            x1 = x1 + res1 if emb1 is not None else None
            x2 = x2 + res2 if emb2 is not None else None
            x3 = x3 + res3 if emb3 is not None else None
            x4 = x4 + res4 if emb4 is not None else None

            return x1, x2, x3, x4


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.layer_norm1 = LayerNorm3d(hidden_features * 2, LayerNorm_type='WithBias')
        self.act = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.layer_norm1(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

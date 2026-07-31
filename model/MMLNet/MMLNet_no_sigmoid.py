# Multi-branch Mutual-guiding Learning for Infrared Small Target Detection

import torch
import torch.nn as nn
import torch.nn.functional as F
# from my_functionals import GatedSpatialConv as gsc
# from network import Resnet
from torch.nn.parameter import Parameter
# from DCNv2.TTOA import TTOA
import pdb
from skimage import measure
import numpy as np
from scipy.io import savemat
import pdb
from pytorch_wavelets import DWTForward



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(BasicConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.basicconv(x)


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, k=4):
        super(DenseLayer, self).__init__()

        self.dense = nn.Sequential(
            nn.Conv2d(in_C, out_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_C),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_C, out_C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dense(x)



# # # # # # # ###################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x + residual, True)
        return out


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class highpass_filter(nn.Module):
    def __init__(self):
        super(highpass_filter, self).__init__()
        self.n = 2
        self.d0 = 0.1

    def forward(self, img):
        # Perform FFT
        s = torch.fft.fftshift(torch.fft.fft2(img))
        batch_size, num_channels, N1, N2 = s.size()

        n1 = N1 // 2
        n2 = N2 // 2

        # Generate frequency grid
        x = torch.linspace(-1, 1, N1).cuda()
        y = torch.linspace(-1, 1, N2).cuda()
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        distance = torch.sqrt(xx ** 2 + yy ** 2)

        # Apply Butterworth highpass filter
        h = 1 / (1 + (distance / self.d0) ** (2 * self.n))
        h = h.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
        h = h.repeat(batch_size, num_channels, 1, 1)
        h = h.cuda()

        # Apply filter in frequency domain
        s_filtered = s * h

        # Perform inverse FFT
        s_inverse = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(s_filtered, dim=(-2, -1)), dim=(-2, -1)))

        return torch.real(s_inverse)


class tensor_dilate(nn.Module):
    def __init__(self, ksize=3):
        super(tensor_dilate, self).__init__()
        self.ksize = ksize

    def forward(self, img):
        # pdb.set_trace()
        B, C, H, W = img.shape
        pad = (self.ksize - 1) // 2
        img = F.pad(img, [pad, pad, pad, pad], mode='constant', value=0)

        patches = img.unfold(dimension=2, size=self.ksize, step=1)
        patches = patches.unfold(dimension=3, size=self.ksize, step=1)

        dilate, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)

        return dilate


class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


'''
Feature Separation Part
'''


class FSP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FSP, self).__init__()
        self.filter = FilterLayer(2 * in_planes, out_planes, reduction)

    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        channel_weight = self.filter(combined)
        out = mainPath + channel_weight * guidePath
        return out


class CCFE(nn.Module):
    def __init__(self):
        super(CCFE, self).__init__()
        self.ccsolver = CCSolver()

    def forward(self, prob, feat):

        B, C, H, W = prob.shape
        r = 2

        feats = []
        for i in range(B):

            labels = self.ccsolver(prob[i].detach())  # numpy
            props = measure.regionprops(labels)
            confidence_values = prob[i].detach().flatten().cpu().numpy()  # numpy

            connected_regions = []
            confidences = []

            if len(np.unique(labels)) - 1 == 0:

                feature = F.interpolate(feat[i].unsqueeze(0), size=(H // r, W // r))
                feats.append(torch.cat([feature, feature, feature], dim=1))
            else:
                region_labels = np.unique(labels)[1:]
                for region_label in region_labels:
                    region_pixels = (labels == region_label).flatten()
                    region_confidences = confidence_values[region_pixels]
                    confidence_mean = np.mean(region_confidences)
                    confidences.append(confidence_mean)

                sorted_indices = np.argsort(confidences)[::-1]
                for index in sorted_indices:
                    region_label = region_labels[index]
                    if region_label not in connected_regions:
                        connected_regions.append(region_label)
                        if len(connected_regions) == min(3, len(np.unique(labels)) - 1):
                            break

                feature = []
                for region_label in connected_regions:
                    min_row, min_col, max_row, max_col = props[region_label - 1].bbox
                    cropped_feature = feat[i][None, :, min_row:max_row, min_col:max_col]
                    cropped_feature = F.interpolate(cropped_feature, size=(H // r, W // r))
                    feature.append(cropped_feature)
                if len(connected_regions) == 1:
                    feature = [feature[0], feature[0], feature[0]]
                elif len(connected_regions) == 2:
                    feature = [feature[0], feature[0], feature[1]]

                feats.append(torch.cat(feature, dim=1))
        feats = torch.cat(feats, dim=0)
        return feats


# 定义图像连通域求解器 ConnectedComponentSolver
class CCSolver(nn.Module):
    def __init__(self):
        super(CCSolver, self).__init__()

    def forward(self, prob):
        binary_image = (prob > 0.5).float()  # 阈值化，将图像转为二值图像
        binary_image = binary_image.squeeze().cpu().numpy()  # 将张量转换为NumPy数组

        # 使用skimage.measure.label函数进行连通域分析
        labels = measure.label(binary_image, connectivity=2)
        return labels


class Unit(nn.Module):
    def __init__(self, n_feats, kernel_size=3, padding=1, bias=False, act=nn.ReLU(inplace=True)):
        super(Unit, self).__init__()

        m = []
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))
        # m.append(nn.BatchNorm2d(n_feats))
        m.append(act)
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=padding, bias=bias))

        self.m = nn.Sequential(*m)

    def forward(self, x):
        x = self.m(x) + x
        return x


class DDPM(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, kernel_size=3, down_factor=4):
        super(DDPM, self).__init__()
        self.kernel_size = kernel_size
        self.mid_c = out_C // 4
        self.down_input = nn.Conv2d(in_xC, self.mid_c, 1)
        self.branch_1 = DepthDC3x3_1(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.branch_3 = DepthDC3x3_3(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.branch_5 = DepthDC3x3_5(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.fuse = BasicConv2d(4 * self.mid_c, out_C, 3, 1, 1)

    def forward(self, x, y):
        x = self.down_input(x)
        result_1 = self.branch_1(x, y)
        result_3 = self.branch_3(x, y)
        result_5 = self.branch_5(x, y)
        return self.fuse(torch.cat((x, result_1, result_3, result_5), dim=1))


class DepthDC3x3_1(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        super(DepthDC3x3_1, self).__init__()
        self.kernel_size = 3
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DepthDC3x3_3(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        super(DepthDC3x3_3, self).__init__()
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.kernel_size = 3
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=3, padding=3, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DepthDC3x3_5(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        super(DepthDC3x3_5, self).__init__()
        self.kernel_size = 3
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=5, padding=5, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


# 改动
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class SobelConv(nn.Module):
    def __init__(self):
        super(SobelConv, self).__init__()
        # 定义 Sobel 核
        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # 拼接 Sobel 核
        self.sobel_kernel = torch.cat([sobel_kernel_x, sobel_kernel_y], dim=0)

    def forward(self, x):
        # 扩展 kernel 以匹配输入的通道数
        sobel_kernel = self.sobel_kernel.repeat(x.size(1), 1, 1, 1).cuda()
        # 进行卷积
        x = F.conv2d(x, sobel_kernel, padding=1, groups=x.size(1))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class CustomModule(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3, out_channels):
        super(CustomModule, self).__init__()
        total_in_channels = in_channels1 + in_channels2 + in_channels3
        self.conv_block = ConvBlock(total_in_channels, out_channels)
        self.sobel_block = SobelConv()
        self.final_conv = nn.Conv2d(166, in_channels1, kernel_size=3, padding=1)

    def forward(self, tensor1, tensor2, tensor3):
        # 拼接输入 tensor
        x = torch.cat((tensor1, tensor2, tensor3), dim=1).cuda()
        # 普通卷积块
        conv_out = self.conv_block(x)
        # conv_out = self.conv_block(conv_out)
        # Sobel 卷积块
        sobel_out = self.sobel_block(x)
        # sobel_out = self.sobel_block(sobel_out)
        # 从普通卷积块出来的结果与 tensor1 相乘
        multiplied_out = conv_out * tensor1
        # 拼接相乘结果与 Sobel 卷积结果
        final_out = torch.cat((multiplied_out, sobel_out), dim=1)
        # print("multiplied_out", multiplied_out.shape) # ([10, 32, 480, 480])
        # print("sobel_out", sobel_out.shape) # ([10, 134, 480, 480])
        # 通过一个卷积层输出
        output = self.final_conv(final_out)
        return output


class CustomModule2(nn.Module):
    def __init__(self, in_channels1, in_channels2, in_channels3, out_channels):
        super(CustomModule2, self).__init__()
        total_in_channels = in_channels1 + in_channels2 + in_channels3
        self.conv_block = ConvBlock(total_in_channels, 134)
        self.sobel_block = SobelConv()
        self.final_conv = nn.Conv2d(134, in_channels1, kernel_size=3, padding=1)

    def forward(self, tensor1, tensor2, tensor3):
        # 拼接输入 tensor
        x = torch.cat((tensor1, tensor2, tensor3), dim=1).cuda()
        # 普通卷积块
        conv_out = self.conv_block(x)
        # conv_out = self.conv_block(conv_out)
        # Sobel 卷积块
        sobel_out = self.sobel_block(x)
        # sobel_out = self.sobel_block(sobel_out)
        # 从普通卷积块出来的结果与 tensor1 相乘
        multiplied_out = conv_out * tensor1
        # 拼接相乘结果与 Sobel 卷积结果
        final_out = multiplied_out + sobel_out
        # print("multiplied_out", multiplied_out.shape) # ([10, 32, 480, 480])
        # print("sobel_out", sobel_out.shape) # ([10, 134, 480, 480])
        # 通过一个卷积层输出
        output = self.final_conv(final_out)
        return output


class MMLNet_No_Sigmoid(nn.Module):
    def __init__(self, layer_blocks=[4] * 3,
                            channels=[16, 32, 64, 128]):
        super(MMLNet_No_Sigmoid, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        # 改动
        self.down1 = Down_wt(16, 16)
        self.down2 = Down_wt(32, 32)
        self.down3 = Down_wt(64, 64)
        self.down4 = Down_wt(32, 64)

        self.stage1 = self._make_layer(block=ResidualBlock, block_num=2,
                                       in_channels=7, out_channels=channels[0], stride=1)
        self.stage1_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[0], out_channels=channels[0], stride=1)
        self.stage1_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[0] * 2, out_channels=channels[0] * 2, stride=1)
        self.stage2_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[0], out_channels=channels[0], stride=1)
        self.stage2_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[0] * 2, out_channels=channels[0] * 2, stride=1)

        self.stage2_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[1], out_channels=channels[1], stride=1)
        self.stage3_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[1], out_channels=channels[1], stride=1)
        self.stage3_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[1] * 2, out_channels=channels[1] * 2, stride=1)
        self.stage2_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[1] * 2, out_channels=channels[1] * 2, stride=1)

        self.stage3_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[2], out_channels=channels[2], stride=1)
        self.stage4_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[2], out_channels=channels[2], stride=1)
        self.stage4_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[2] * 2, out_channels=channels[2] * 2, stride=1)
        self.stage3_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                         in_channels=channels[2] * 2, out_channels=channels[2] * 2, stride=1)

        self.edge_stage1 = self._make_layer(block=ResidualBlock, block_num=2,
                                            in_channels=3, out_channels=channels[0], stride=1)
        self.edge_stage1_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[0], out_channels=channels[0], stride=1)
        self.edge_stage1_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[0] * 2, out_channels=channels[0] * 2, stride=1)
        self.edge_stage2_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[0], out_channels=channels[0], stride=1)
        self.edge_stage2_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[0] * 2, out_channels=channels[0] * 2, stride=1)

        self.edge_stage2_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[1], out_channels=channels[1], stride=1)
        self.edge_stage3_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[1], out_channels=channels[1], stride=1)
        self.edge_stage3_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[1] * 2, out_channels=channels[1] * 2, stride=1)
        self.edge_stage2_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[1] * 2, out_channels=channels[1] * 2, stride=1)

        self.edge_stage3_3 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[2], out_channels=channels[2], stride=1)
        self.edge_stage4_1 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[2], out_channels=channels[2], stride=1)
        self.edge_stage4_2 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[2] * 2, out_channels=channels[2] * 2, stride=1)
        self.edge_stage3_4 = self._make_layer(block=ResidualBlock, block_num=1,
                                              in_channels=channels[2] * 2, out_channels=channels[2] * 2, stride=1)

        self.uplayer2 = self._make_layer(block=ResidualBlock, block_num=2,
                                         in_channels=channels[3], out_channels=channels[3], stride=1)

        self.uplayer2_edge = self._make_layer(block=ResidualBlock, block_num=2,
                                              in_channels=channels[3], out_channels=channels[3], stride=1)

        self.uplayer1 = self._make_layer(block=ResidualBlock, block_num=2,
                                         in_channels=channels[2], out_channels=channels[2], stride=1)

        self.uplayer1_edge = self._make_layer(block=ResidualBlock, block_num=2,
                                              in_channels=channels[2], out_channels=channels[2], stride=1)

        self.dilate = tensor_dilate()

        sobel_x = torch.tensor([[[1, 0], [0, -1]],
                                [[1, 0], [0, -1]],
                                [[1, 0], [0, -1]]], dtype=torch.float32)
        sobel_x = sobel_x.reshape(1, 3, 2, 2).cuda()
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False).cuda()

        sobel_y = torch.tensor([[[0, 1], [-1, 0]],
                                [[0, 1], [-1, 0]],
                                [[0, 1], [-1, 0]]], dtype=torch.float32)
        sobel_y = sobel_y.reshape(1, 3, 2, 2).cuda()
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False).cuda()

        self.highpass_filter = highpass_filter()
        self.sigmoid = nn.Sigmoid()

        self.fsp_rgb = FSP(128, 128, reduction=16)
        self.fsp_hha = FSP(128, 128, reduction=16)
        self.fsp_rgb1 = FSP(64, 64, reduction=8)
        self.fsp_hha1 = FSP(64, 64, reduction=8)

        self.fuse = nn.Conv2d(32, 1, kernel_size=1, padding=0, bias=False)
        self.reduce2_edge = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.reduce1_edge = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.reduce2 = nn.Conv2d(128, 64, kernel_size=1, padding=0, bias=False)
        self.reduce1 = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.add1 = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False)

        self.ccfe = CCFE()

        self.fuse_local = nn.Conv2d(96, 128, kernel_size=1, padding=0, bias=False)
        self.upsample = nn.PixelShuffle(2)

        self.unit = Unit(32)
        self.end = _FCNHead(32, 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.max_pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=int((3 - 1) / 2))  # 可调整kernel_size
        self.max_pool5 = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=int((5 - 1) / 2))  # 可调整kernel_size
        self.max_pool7 = torch.nn.MaxPool2d(kernel_size=7, stride=1, padding=int((7 - 1) / 2))  # 可调整kernel_size

        self.stage_fuse1 = self._make_layer(block=ResidualBlock, block_num=2,
                                            in_channels=9, out_channels=3, stride=1)
        self.stage_fuse2 = self._make_layer(block=ResidualBlock, block_num=2,
                                            in_channels=3, out_channels=3, stride=1)
        self.stage_fuse3 = self._make_layer(block=ResidualBlock, block_num=2,
                                            in_channels=3, out_channels=1, stride=1)

        self.custom = CustomModule(in_channels1=32, in_channels2=32, in_channels3=3, out_channels=32).cuda()

    def forward(self, x):
        _, _, hei, wid = x.shape  ##0
        x_size = x.size()
        x_dilate = self.dilate(x)
        # pdb.set_trace()

        # 改动 膨胀腐蚀先验 ######################
        # print("x", x)
        # print("x", x.shape) # ([8, 3, 480, 480])
        # tensor_erode3 = -self.max_pool3(-x)
        tensor_open3 = self.max_pool3(x)
        # tensor_dilate_value3 = self.max_pool3(x)
        # tensor_close3 = -self.max_pool3(-tensor_dilate_value3)
        tensor_tophot3 = x - tensor_open3

        # tensor_erode5 = -self.max_pool5(-x)
        tensor_open5 = self.max_pool5(x)
        # tensor_dilate_value5 = self.max_pool5(x)
        # tensor_close5 = -self.max_pool5(-tensor_dilate_value5)
        tensor_tophot5 = x - tensor_open5

        # tensor_erode7 = -self.max_pool7(-x)
        tensor_open7 = self.max_pool7(x)
        # tensor_dilate_value7 = self.max_pool7(x)
        # tensor_close7 = -self.max_pool7(-tensor_dilate_value7)
        tensor_tophot7 = x - tensor_open7

        pred_mul = torch.cat([tensor_tophot3, tensor_tophot5, tensor_tophot7], 1)
        pred_mul1 = self.stage_fuse1(pred_mul)
        pred_mul2 = self.stage_fuse2(pred_mul1)
        pred_mul3 = self.stage_fuse3(pred_mul2)

        # print("pred_mul", pred_mul.shape) #

        #####################################

        # savemat('1.mat', {'x_dilate': x_dilate[0,].cpu().numpy().transpose(1, 2, 0),
        #                   'x': x[0,].cpu().numpy().transpose(1, 2, 0)})

        Gx = F.conv2d(x_dilate, self.weight_x, padding='same')
        Gy = F.conv2d(x_dilate, self.weight_y, padding='same')
        x_gard = torch.sqrt(Gx ** 2 + Gy ** 2)

        stage1 = self.stage1(torch.cat([x, x_gard, x_dilate], 1))  # 16x480x480

        fisrt = self.stage1_1(stage1)  # 16x480x480

        second = self.stage2_1(self.pool(stage1))  # 16x240x240
        third = self.stage2_2(torch.cat([self.down1(fisrt), second], 1))  # 32x240x240
        stage1_2 = self.stage1_2(torch.cat([self.up(second), fisrt], 1))  # 32x480x480

        fisrt = self.stage2_3(third)  # 32x240x240
        second = self.stage3_1(self.pool(third))  # 32x120x120
        third = self.stage3_2(torch.cat([self.down2(fisrt), second], 1))  # 64x120x120
        stage2_4 = self.stage2_4(torch.cat([self.up(second), fisrt], 1))  # 64x240x240

        fisrt = self.stage3_3(third)  # 64x120x120
        second = self.stage4_1(self.pool(third))  # 64x60x60
        stage4_2 = self.stage4_2(torch.cat([self.down3(fisrt), second], 1))  # 128x60x60
        stage3_4 = self.stage3_4(torch.cat([self.up(second), fisrt], 1))  # 128x120x120

        stage1 = self.edge_stage1(x)  # 16x480x480

        fisrt = self.edge_stage1_1(stage1)  # 16x480x480
        second = self.edge_stage2_1(self.pool(stage1))  # 16x240x240
        third = self.edge_stage2_2(torch.cat([self.down(fisrt), second], 1))  # 32x240x240
        # _ = self.edge_stage1_2(torch.cat([self.up(second), fisrt],1)) # 32x480x480

        fisrt = self.edge_stage2_3(third)  # 32x240x240
        second = self.edge_stage3_1(self.pool(third))  # 32x120x120
        third = self.edge_stage3_2(torch.cat([self.down(fisrt), second], 1))  # 64x120x120
        edge_stage2_4 = self.edge_stage2_4(torch.cat([self.up(second), fisrt], 1))  # 64x240x240

        fisrt = self.edge_stage3_3(third)  # 64x120x120
        second = self.edge_stage4_1(self.pool(third))  # 64x60x60
        edge_stage4_2 = self.edge_stage4_2(torch.cat([self.down(fisrt), second], 1))  # 128x60x60
        edge_stage3_4 = self.edge_stage3_4(torch.cat([self.up(second), fisrt], 1))  # 128x120x120

        rec_deconc = self.fsp_rgb(edge_stage4_2, stage4_2)  ##128x60x60
        rec_edge = self.fsp_hha(stage4_2, edge_stage4_2)  ##128x60x60

        rec_deconc = self.uplayer2(rec_deconc)  ##128x60x60
        rec_edge = self.uplayer2_edge(rec_edge)  ##128x60x60

        rec_edge = self.reduce2_edge(self.up(rec_edge) + edge_stage3_4)  ##    64x120x120
        # 改动
        # rec_deconc = self.reduce2(self.up(rec_deconc) + stage3_4)  ## 64x120x120
        # print("1", self.down2(self.down2(stage1_2)).shape) # ([10, 32, 120, 120])
        # print("2", self.down3(stage2_4).shape) # ([10, 64, 120, 120])
        # print("3", self.reduce2(self.up(rec_deconc) + stage3_4).shape)
        rec_deconc = self.reduce2(self.up(rec_deconc) + stage3_4) + self.down3(stage2_4) + self.down4(
            self.down2(stage1_2))  ## 64x120x120

        rec_deconc = self.fsp_rgb1(rec_edge, rec_deconc)  ##64x120x120
        rec_edge = self.fsp_hha1(rec_deconc, rec_edge)  ##64x120x120

        rec_deconc = self.uplayer1(rec_deconc)  ##64x120x120
        rec_edge = self.uplayer1_edge(rec_edge)  ##64x120x120

        rec_edge = self.reduce1_edge(self.up(rec_edge) + edge_stage2_4)  ##   32x240x240
        # 未改动
        # rec_deconc = self.reduce1(self.up(rec_deconc) + stage2_4)  ##    32x240x240
        # rec_deconc = self.reduce1(self.up(rec_deconc) + stage2_4)  + self.down2(stage1_2)  ##    32x240x240
        rec_deconc = self.reduce1(self.up(rec_deconc) + stage2_4) + self.down2((stage1_2)) + self.reduce1(
            self.reduce2(self.up(stage3_4)))

        cs = F.interpolate(rec_edge, x_size[2:],
                           mode='bilinear', align_corners=True)
        cs = self.fuse(cs)
        edge_out = self.sigmoid(cs)

        rec_deconc = F.interpolate(rec_deconc, size=[hei, wid], mode='bilinear')

        diff_edge = self.ccfe(edge_out, rec_deconc)
        diff_edge = self.fuse_local(diff_edge)
        diff_edge = self.upsample(diff_edge)

        # print("diff_edge", diff_edge.shape) # ([10, 32, 480, 480])
        # print("rec_deconc", rec_deconc.shape) # ([10, 32, 480, 480])
        # print("pred_mul2", pred_mul2.shape) # ([10, 3, 480, 480])
        # pred = torch.cat([diff_edge, rec_deconc, pred_mul2], 1) # 10,67,480,480
        pred = self.custom(diff_edge, rec_deconc, pred_mul2)

        pred = self.unit(pred)
        pred = self.end(pred)

        # return pred, edge_out
        return pred


    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)


if __name__ == '__main__':
    # net = MMLNet_No_Sigmoid(layer_blocks=[4] * 3,
    #             channels=[16, 32, 64, 128])
    net = MMLNet_No_Sigmoid()
    print(net)
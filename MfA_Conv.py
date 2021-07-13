import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MfA_Conv_start(nn.Module):
    def __init__(self, in_channels, out_channels, high_L=3, high_S=1, low_L=5, low_S=3, high_stride=1, low_stride=1,
                 basic_stride=1, padding=1, bias=True):
        """
        :param in_channels:
        :param out_channels:
        :param high_L:
        :param high_S:
        :param low_L:
        :param low_S:
        :param high_stride:
        :param low_stride:
        :param basic_stride:
        :param padding: padding on low and long conv
        """
        super(MfA_Conv_start, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.high_L = high_L
        self.high_S = high_S
        self.low_L = low_L
        self.low_S = low_S
        self.high_stride = high_stride
        self.low_stride = low_stride
        self.basic_stride = basic_stride
        self.padding_hl, self.padding_hs, self.padding_ll, self.padding_ls = self.count_padding()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        # 改进思路 不用全部in_channel用来卷，可以份一半，一半用来分析高频，一半用来分析低频
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(high_S, high_L),
                                        stride=(high_stride, basic_stride), padding=(self.padding_hs, self.padding_hl))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(high_L, high_S),
                                        stride=(basic_stride, high_stride), padding=(self.padding_hl, self.padding_hs))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(low_S, low_L),
                                       stride=(low_stride, basic_stride), padding=(self.padding_ls, self.padding_ll))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(low_L, low_S),
                                       stride=(low_stride, basic_stride), padding=(self.padding_ll, self.padding_ls))
        # self.weight_hh = nn.Parameter(torch.FloatTensor(2))
        # self.weight_hh.data = F.softmax(self.weight_hh, dim=0)  # attention this weight.data
        # self.weight_ll = nn.Parameter(torch.FloatTensor(2))
        # self.weight_ll.data = F.softmax(self.weight_ll, dim=0)
        # self.weight_hl = nn.Parameter(torch.FloatTensor(2))
        # self.weight_hl.data = F.softmax(self.weight_hl, dim=0)
        # self.weight_lh = nn.Parameter(torch.FloatTensor(2))
        # self.weight_lh.data = F.softmax(self.weight_lh, dim=0)
        # self.weight = nn.Parameter(torch.FloatTensor(4))
        # self.weight.data = F.softmax(self.weight, dim=0)  # attention this weight.data
        # self.alpha, self.beta, self.delta, self.gamma = self.get_weight()

    def count_padding(self):
        padding_hl = self.high_L // 2
        padding_hs = self.high_S // 2
        padding_ll = self.low_L // 2
        padding_ls = self.low_S // 2
        return padding_hl, padding_hs, padding_ll, padding_ls

    def get_weight(self):
        weight = nn.Parameter(torch.FloatTensor(4))
        weight.data = F.softmax(weight, dim=0)     # attention this weight.data
        alpha = weight[0]
        beta = weight[1]
        delta = weight[2]
        gamma = weight[3]
        return alpha, beta, delta, gamma

    def feature_map_add(self, weight_1, weight_2, feature_map_1, feature_map_2):
        up_1_size = self.out_channel // 3
        up_2_size = up_1_size + self.out_channel // 3
        up_1_feature = weight_1 * 0.75 * feature_map_1[:, 0:up_1_size, :, :] + \
                       weight_2 * 0.25 * feature_map_2[:, 0:up_1_size, :, :]
        up_2_feature = weight_1 * 0.25 * feature_map_1[:, up_1_size:up_2_size, :, :] + \
                       weight_2 * 0.75 * feature_map_2[:, up_1_size:up_2_size, :, :]
        equal_feature = weight_1 * 0.5 * feature_map_1[:, up_2_size:self.out_channel, :, :] + \
                        weight_2 * 0.5 * feature_map_2[:, up_2_size:self.out_channel, :, :]
        final_out = torch.cat([up_1_feature, up_2_feature, equal_feature], 1)
        return final_out


    def forward(self, x):
        # x_1 = self.downsample(x)
        out_Horn_h = self.Hori_high_conv(x)
        out_Vert_h = self.Vert_high_conv(x)
        out_Horn_l = self.Hori_low_conv(x)
        # out_Horn_l = F.interpolate(out_Horn_l, scale_factor=2)
        out_Vert_l = self.Vert_low_conv(x)
        # out_Vert_l = F.interpolate(out_Vert_l, scale_factor=2)
        # self.weight_hh.data = F.softmax(self.weight_hh, dim=0)
        # out_hh = self.weight_hh[0] * out_Vert_h + self.weight_hh[1] * out_Horn_h
        # self.weight_ll.data = F.softmax(self.weight_ll, dim=0)
        # out_ll = self.weight_ll[0] * out_Vert_l + self.weight_ll[1] * out_Horn_l
        # self.weight_hl.data = F.softmax(self.weight_hl, dim=0)
        # out_hl = self.weight_hl[0] * out_Vert_h + self.weight_hl[1] * out_Horn_l
        # self.weight_lh.data = F.softmax(self.weight_lh, dim=0)
        # out_lh = self.weight_lh[0] * out_Vert_l + self.weight_lh[1] * out_Horn_h
        # out_hh = self.feature_map_add(self.weight[0], self.weight[1], out_Vert_h, out_Horn_h)
        # out_ll = self.feature_map_add(self.weight[2], self.weight[3], out_Vert_l, out_Horn_l)
        # out_hl = self.feature_map_add(self.weight[0], self.weight[3], out_Vert_h, out_Horn_l)
        # out_lh = self.feature_map_add(self.weight[2], self.weight[1], out_Vert_l, out_Horn_h)
        out_hh = 0.5 * out_Vert_h + 0.5 * out_Horn_h
        out_ll = 0.5 * out_Vert_l + 0.5 * out_Horn_l
        out_hl = 0.5 * out_Vert_h + 0.5 * out_Horn_l
        out_lh = 0.5 * out_Vert_l + 0.5 * out_Horn_h
        final_out = torch.cat([out_hh, out_ll, out_hl, out_lh], 1)
        return final_out


class MfA_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, high_L=3, high_S=1, low_L=5, low_S=3, high_stride=1, low_stride=1,
                 basic_stride=1, padding=1, bias=True):
        """
        :param in_channels:
        :param out_channels:
        :param high_L:
        :param high_S:
        :param low_L:
        :param low_S:
        :param high_stride:
        :param low_stride:
        :param basic_stride:
        :param padding: padding on low and long conv
        """
        super(MfA_Conv, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.high_L = high_L
        self.high_S = high_S
        self.low_L = low_L
        self.low_S = low_S
        self.high_stride = high_stride
        self.low_stride = low_stride
        self.basic_stride = basic_stride
        self.padding_hl, self.padding_hs, self.padding_ll, self.padding_ls = self.count_padding()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        # 改进思路 不用全部in_channel用来卷，可以份一半，一半用来分析高频，一半用来分析低频
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=(high_S, high_L),
                                        stride=(high_stride, basic_stride), padding=(self.padding_hs, self.padding_hl))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=(high_L, high_S),
                                        stride=(basic_stride, high_stride), padding=(self.padding_hl, self.padding_hs))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=(low_S, low_L),
                                       stride=(low_stride, basic_stride), padding=(self.padding_ls, self.padding_ll))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels, kernel_size=(low_L, low_S),
                                       stride=(low_stride, basic_stride), padding=(self.padding_ll, self.padding_ls))
        # self.weight_hh = nn.Parameter(torch.FloatTensor(2))
        # self.weight_hh.data = F.softmax(self.weight_hh, dim=0)  # attention this weight.data
        # self.weight_ll = nn.Parameter(torch.FloatTensor(2))
        # self.weight_ll.data = F.softmax(self.weight_ll, dim=0)
        # self.weight_hl = nn.Parameter(torch.FloatTensor(2))
        # self.weight_hl.data = F.softmax(self.weight_hl, dim=0)
        # self.weight_lh = nn.Parameter(torch.FloatTensor(2))
        # self.weight_lh.data = F.softmax(self.weight_lh, dim=0)
        # self.weight = nn.Parameter(torch.FloatTensor(4))
        # self.weight.data = F.softmax(self.weight, dim=0)  # attention this weight.data
        # self.alpha, self.beta, self.delta, self.gamma = self.get_weight()

    def count_padding(self):
        padding_hl = self.high_L // 2
        padding_hs = self.high_S // 2
        padding_ll = self.low_L // 2
        padding_ls = self.low_S // 2
        return padding_hl, padding_hs, padding_ll, padding_ls

    def feature_map_add(self, weight_1, weight_2, feature_map_1, feature_map_2):
        up_1_size = self.out_channel // 3
        up_2_size = up_1_size + self.out_channel // 3
        up_1_feature = weight_1 * 0.75 * feature_map_1[:, 0:up_1_size, :, :] + \
                       weight_2 * 0.25 * feature_map_2[:, 0:up_1_size, :, :]
        up_2_feature = weight_1 * 0.25 * feature_map_1[:, up_1_size:up_2_size, :, :] + \
                       weight_2 * 0.75 * feature_map_2[:, up_1_size:up_2_size, :, :]
        equal_feature = weight_1 * 0.5 * feature_map_1[:, up_2_size:self.out_channel, :, :] + \
                       weight_2 * 0.5 * feature_map_2[:, up_2_size:self.out_channel, :, :]
        final_out = torch.cat([up_1_feature, up_2_feature, equal_feature], 1)
        return final_out

    def forward(self, x):
        x_Horn_h = torch.cat([x[:, 0:self.in_channel//4, :, :], x[:, self.in_channel*3//4:self.in_channel*4//4, :, :]], 1)
        out_Horn_h = self.Hori_high_conv(x_Horn_h)

        x_Vert_h = torch.cat([x[:, 0:self.in_channel//4, :, :], x[:, self.in_channel//2:self.in_channel*3//4, :, :]], 1)
        out_Vert_h = self.Vert_high_conv(x_Vert_h)

        x_Horn_l = torch.cat([x[:, self.in_channel//4:self.in_channel//2, :, :], x[:, self.in_channel//2:self.in_channel*3//4, :, :]], 1)
        # x_Horn_l = self.downsample(x_Horn_l)
        out_Horn_l = self.Hori_low_conv(x_Horn_l)
        # out_Horn_l = F.interpolate(out_Horn_l, scale_factor=2)

        x_Vert_l = torch.cat([x[:, self.in_channel//4:self.in_channel//2, :, :], x[:, self.in_channel*3//4:self.in_channel*4//4, :, :]], 1)
        # x_Vert_l = self.downsample(x_Vert_l)
        out_Vert_l = self.Vert_low_conv(x_Vert_l)
        # out_Vert_l = F.interpolate(out_Vert_l, scale_factor=2)

        # self.weight_hh.data = F.softmax(self.weight_hh, dim=0)
        # out_hh = self.weight_hh[0] * out_Vert_h + self.weight_hh[1] * out_Horn_h
        # self.weight_ll.data = F.softmax(self.weight_ll, dim=0)
        # out_ll = self.weight_ll[0] * out_Vert_l + self.weight_ll[1] * out_Horn_l
        # self.weight_hl.data = F.softmax(self.weight_hl, dim=0)
        # out_hl = self.weight_hl[0] * out_Vert_h + self.weight_hl[1] * out_Horn_l
        # self.weight_lh.data = F.softmax(self.weight_lh, dim=0)
        # out_lh = self.weight_lh[0] * out_Vert_l + self.weight_lh[1] * out_Horn_h
        # out_hh = self.feature_map_add(self.weight[0], self.weight[1], out_Vert_h, out_Horn_h)
        # out_ll = self.feature_map_add(self.weight[2], self.weight[3], out_Vert_l, out_Horn_l)
        # out_hl = self.feature_map_add(self.weight[0], self.weight[3], out_Vert_h, out_Horn_l)
        # out_lh = self.feature_map_add(self.weight[2], self.weight[1], out_Vert_l, out_Horn_h)
        out_hh = 0.5 * out_Vert_h + 0.5 * out_Horn_h
        out_ll = 0.5 * out_Vert_l + 0.5 * out_Horn_l
        out_hl = 0.5 * out_Vert_h + 0.5 * out_Horn_l
        out_lh = 0.5 * out_Vert_l + 0.5 * out_Horn_h
        final_out = torch.cat([out_hh, out_ll, out_hl, out_lh], 1)
        return final_out


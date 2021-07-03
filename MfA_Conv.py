import torch
from torch import nn
import numpy as np

class MfA_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, high_L=3, high_S=1, low_L=5, low_S=3, high_stride=1, low_stride=1, basic_stride=1, padding=1):
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
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(high_S, high_L),
                                        stride=(high_stride, basic_stride))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(high_L, high_S),
                                        stride=(basic_stride, high_stride))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(low_S, low_L),
                                       stride=(low_stride, basic_stride))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(low_L, low_S),
                                       stride=(low_stride, basic_stride))

    def count_padding(self):
        padding_hl = self.high_L // 2
        padding_hs = self.high_S // 2
        padding_ll = self.low_L // 2
        padding_ls = self.low_S // 2
        return padding_hl, padding_hs, padding_ll, padding_ls

    def get_weight(self):
        alpha = 0.25
        beta = 0.25
        delta = 0.25
        gamma = 0.25
        return alpha, beta, delta, gamma




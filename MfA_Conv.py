import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

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
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(high_S, high_L),
                                        stride=(high_stride, basic_stride), padding=(self.padding_hs, self.padding_hl))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(high_L, high_S),
                                        stride=(basic_stride, high_stride), padding=(self.padding_hl, self.padding_hs))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(low_S, low_L),
                                       stride=(low_stride, basic_stride), padding=(self.padding_ls, self.padding_ll))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(low_L, low_S),
                                       stride=(low_stride, basic_stride), padding=(self.padding_ll, self.padding_ls))
        self.alpha, self.beta, self.delta, self.gamma = self.get_weight()

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

    def forward(self, x):
        out_Horn_h = self.Hori_high_conv(x)
        out_Vert_h = self.Vert_high_conv(x)
        out_Horn_l = self.Hori_low_conv(x)
        out_Vert_l = self.Vert_low_conv(x)
        # new_alpha, new_beta = self.alpha / (self.alpha + self.beta), self.beta / (self.alpha + self.beta)
        # out_hh = new_alpha * out_Vert_h + new_beta * out_Horn_h
        # new_delta, new_gamma = self.delta / (self.delta + self.gamma), self.gamma / (self.delta + self.gamma)
        # out_ll = new_delta * out_Vert_l + new_gamma * out_Horn_l
        # new_alpha, new_gamma = self.alpha / (self.alpha + self.gamma), self.gamma / (self.alpha + self.gamma)
        # out_hl = new_alpha * out_Vert_h + new_gamma * out_Horn_l
        # new_delta, new_beta = self.delta / (self.delta + self.beta), self.beta / (self.delta + self.beta)
        # out_lh = new_delta * out_Vert_l + new_beta * out_Horn_h

        out_hh = self.alpha * out_Vert_h + self.beta * out_Horn_h
        out_ll = self.delta * out_Vert_l + self.gamma * out_Horn_l
        out_hl = self.alpha * out_Vert_h + self.gamma * out_Horn_l
        out_lh = self.delta * out_Vert_l + self.beta * out_Horn_h
        final_out = torch.cat([out_hh, out_ll, out_hl, out_lh], 1)
        return final_out




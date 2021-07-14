import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MfA_layer1(nn.Module):
    def __init__(self, in_channels, out_channels, high_L=3, high_S=1, low_L=5, low_S=3, stride=1):
        super(MfA_layer1, self).__init__()
        # In fact, the output is twice that of out_channels
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=(high_S, high_L),
                                        stride=stride, padding=(high_S // 2, high_L // 2))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=(high_L, high_S),
                                        stride=stride, padding=(high_L // 2, high_S // 2))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=(low_S, low_L),
                                       stride=stride, padding=(low_S // 2, low_L // 2))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//4, kernel_size=(low_L, low_S),
                                       stride=stride, padding=(low_L // 2, low_S // 2))

    def forward(self, x):
        out_Horn_h = self.Hori_high_conv(x)
        out_Vert_h = self.Vert_high_conv(x)
        out_Horn_l = self.Hori_low_conv(x)
        out_Vert_l = self.Vert_low_conv(x)
        out_hh = 0.5 * out_Vert_h + 0.5 * out_Horn_h
        out_ll = 0.5 * out_Vert_l + 0.5 * out_Horn_l
        out_hl = 0.5 * out_Vert_h + 0.5 * out_Horn_l
        out_lh = 0.5 * out_Vert_l + 0.5 * out_Horn_h
        final_out = torch.cat([out_hh, out_ll, out_hl, out_lh], 1)
        return final_out


class MfA_conv_start(nn.Module):
    def __init__(self, in_channels, out_channels, high_L=3, high_S=1, low_L=5, low_S=3, stride=1):
        super(MfA_conv_start, self).__init__()
        # In fact, the output is twice that of out_channels
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//2, kernel_size=(high_S, high_L),
                                        stride=stride, padding=(high_S // 2, high_L // 2))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//2, kernel_size=(high_L, high_S),
                                        stride=stride, padding=(high_L // 2, high_S // 2))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//2, kernel_size=(low_S, low_L),
                                       stride=stride, padding=(low_S // 2, low_L // 2))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//2, kernel_size=(low_L, low_S),
                                       stride=stride, padding=(low_L // 2, low_S // 2))

    def forward(self, x):
        x_Horn_h = torch.cat(
            [x[:, 0:self.in_channel // 4, :, :], x[:, self.in_channel * 3 // 4:self.in_channel * 4 // 4, :, :]], 1)
        out_Horn_h = self.Hori_high_conv(x_Horn_h)

        x_Vert_h = torch.cat(
            [x[:, 0:self.in_channel // 4, :, :], x[:, self.in_channel // 2:self.in_channel * 3 // 4, :, :]], 1)
        out_Vert_h = self.Vert_high_conv(x_Vert_h)

        x_Horn_l = torch.cat([x[:, self.in_channel // 4:self.in_channel // 2, :, :],
                              x[:, self.in_channel // 2:self.in_channel * 3 // 4, :, :]], 1)
        out_Horn_l = self.Hori_low_conv(x_Horn_l)

        x_Vert_l = torch.cat([x[:, self.in_channel // 4:self.in_channel // 2, :, :],
                              x[:, self.in_channel * 3 // 4:self.in_channel * 4 // 4, :, :]], 1)
        out_Vert_l = self.Vert_low_conv(x_Vert_l)
        out_hh = 0.25 * out_Vert_h + 0.75 * out_Horn_h
        out_hl = 0.75 * out_Vert_h + 0.25 * out_Horn_l
        out_ll = 0.25 * out_Vert_l + 0.75 * out_Horn_l
        out_lh = 0.75 * out_Vert_l + 0.25 * out_Horn_h
        final_out = torch.cat([out_hh, out_hl, out_ll, out_lh], 1)
        return final_out


class MfA_conv(nn.Module):
    def __init__(self, in_channels, out_channels, high_L=3, high_S=1, low_L=5, low_S=3, stride=1):
        super(MfA_conv, self).__init__()
        # In fact, the output is twice that of out_channels
        # 这个模块使用时in_channels应该刚好是out_channels
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels//4, out_channels=out_channels//2, kernel_size=(high_S, high_L),
                                        stride=stride, padding=(high_S // 2, high_L // 2))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels//4, out_channels=out_channels//2, kernel_size=(high_L, high_S),
                                        stride=stride, padding=(high_L // 2, high_S // 2))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels//4, out_channels=out_channels//2, kernel_size=(low_S, low_L),
                                       stride=stride, padding=(low_S // 2, low_L // 2))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels//4, out_channels=out_channels//2, kernel_size=(low_L, low_S),
                                       stride=stride, padding=(low_L // 2, low_S // 2))

    def forward(self, x):
        x_Horn_h = x[:, 0:self.in_channel // 4, :, :]
        x_Vert_h = x[:, self.in_channel // 4:self.in_channel // 2, :, :]
        x_Horn_l = x[:, self.in_channel // 2:self.in_channel * 3 // 4, :, :]
        x_Vert_l = x[:, self.in_channel * 3 // 4:self.in_channel * 4 // 4, :, :]
        out_Horn_h = self.Hori_high_conv(x_Horn_h)
        out_Vert_h = self.Vert_high_conv(x_Vert_h)
        out_Horn_l = self.Hori_low_conv(x_Horn_l)
        out_Vert_l = self.Vert_low_conv(x_Vert_l)
        out_hh = 0.5 * out_Vert_h + 0.5 * out_Horn_h
        out_hl = 0.5 * out_Vert_h + 0.5 * out_Horn_l
        out_ll = 0.5 * out_Vert_l + 0.5 * out_Horn_l
        out_lh = 0.5 * out_Vert_l + 0.5 * out_Horn_h
        final_out = torch.cat([out_hh, out_ll, out_hl, out_lh], 1)
        return final_out


class MfA_conv_end(nn.Module):
    def __init__(self, in_channels, out_channels, high_L=3, high_S=1, low_L=5, low_S=3, stride=1):
        super(MfA_conv_end, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.Hori_high_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//4, kernel_size=(high_S, high_L),
                                        stride=stride, padding=(high_S // 2, high_L // 2))
        self.Vert_high_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//4, kernel_size=(high_L, high_S),
                                        stride=stride, padding=(high_L // 2, high_S // 2))
        self.Hori_low_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//4, kernel_size=(low_S, low_L),
                                       stride=stride, padding=(low_S // 2, low_L // 2))
        self.Vert_low_conv = nn.Conv2d(in_channels=in_channels//2, out_channels=out_channels//4, kernel_size=(low_L, low_S),
                                       stride=stride, padding=(low_L // 2, low_S // 2))

    def forward(self, x):
        x_Horn_h = torch.cat(
            [x[:, 0:self.in_channel // 4, :, :], x[:, self.in_channel * 3 // 4:self.in_channel * 4 // 4, :, :]], 1)
        out_Horn_h = self.Hori_high_conv(x_Horn_h)

        x_Vert_h = torch.cat(
            [x[:, 0:self.in_channel // 4, :, :], x[:, self.in_channel // 2:self.in_channel * 3 // 4, :, :]], 1)
        out_Vert_h = self.Vert_high_conv(x_Vert_h)

        x_Horn_l = torch.cat([x[:, self.in_channel // 4:self.in_channel // 2, :, :],
                              x[:, self.in_channel // 2:self.in_channel * 3 // 4, :, :]], 1)
        out_Horn_l = self.Hori_low_conv(x_Horn_l)

        x_Vert_l = torch.cat([x[:, self.in_channel // 4:self.in_channel // 2, :, :],
                              x[:, self.in_channel * 3 // 4:self.in_channel * 4 // 4, :, :]], 1)
        out_Vert_l = self.Vert_low_conv(x_Vert_l)
        out_hh = 0.5 * out_Vert_h + 0.5 * out_Horn_h
        out_hl = 0.5 * out_Vert_h + 0.5 * out_Horn_l
        out_ll = 0.5 * out_Vert_l + 0.5 * out_Horn_l
        out_lh = 0.5 * out_Vert_l + 0.5 * out_Horn_h
        final_out = torch.cat([out_hh, out_hl, out_ll, out_lh], 1)
        return final_out


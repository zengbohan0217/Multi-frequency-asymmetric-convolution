from torch import nn
from MfA_Conv import MfA_Conv

class ConvBlock(nn.Module):
    def __init__(self, in_channel, kernels, filters, s):
        """
        :param in_channel:
        :param kernels: a list: kernel size
        :param filters: a list: feature maps channel number,
                        Each value should be a quarter of its normal value
        :param s: stride
        """
        super(ConvBlock, self).__init__()
        high_l, high_s, low_l, low_s = kernels
        f1, f2, f3 = filters
        self.stage = nn.Sequential(
            MfA_Conv(in_channel, f1, high_stride=s, low_stride=s, basic_stride=s),
            nn.BatchNorm2d(f1*4),
            nn.ReLU(True),
            MfA_Conv(f1*4, f2, high_L=high_l, high_S=high_s, low_L=low_l, low_S=low_s),
            nn.BatchNorm2d(f2*4),
            nn.ReLU(True),
            MfA_Conv(f2*4, f3),
            nn.BatchNorm2d(f3*4),
        )
        self.shortcut_1 = MfA_Conv(in_channel, f3, high_stride=s, low_stride=s, basic_stride=s)
        self.batch_1 = nn.BatchNorm2d(f3*4)
        self.relu_1 = nn.ReLU(True)

    def forward(self, x):
        x_shortcut = self.shortcut_1(x)
        x_shortcut = self.batch_1(x_shortcut)
        x = self.stage(x)
        x = x + x_shortcut
        x = self.relu_1(x)
        return x

class IndentityBlock(nn.Module):
    def __init__(self, in_channel, kernels, filters):
        super(IndentityBlock, self).__init__()
        high_l, high_s, low_l, low_s = kernels
        f1, f2, f3 = filters
        self.stage = nn.Sequential(
            MfA_Conv(in_channel, f1),
            nn.BatchNorm2d(f1*4),
            nn.ReLU(True),
            MfA_Conv(f1*4, f2, high_L=high_l, high_S=high_s, low_L=low_l, low_S=low_s),
            nn.BatchNorm2d(f2*4),
            nn.ReLU(True),
            MfA_Conv(f2*4, f3),
            nn.BatchNorm2d(f3*4)
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, x):
        x_shortcut = x
        x = self.stage(x)
        x = x + x_shortcut
        x = self.relu_1(x)
        return x




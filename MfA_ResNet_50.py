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

class ResModule(nn.Module):
    def __init__(self, in_channels, n_class):
        super(ResModule, self).__init__()
        self.stage1 = nn.Sequential(
            MfA_Conv(in_channels, 16, high_L=5, high_S=3, low_L=7, low_S=5,
                     basic_stride=2, high_stride=2, low_stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, kernels=[5, 1, 5, 3], filters=[16, 16, 64], s=1),
            IndentityBlock(256, kernels=[5, 1, 5, 3], filters=[16, 16, 64]),
            IndentityBlock(256, kernels=[5, 1, 5, 3], filters=[16, 16, 64]),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(256, kernels=[5, 1, 5, 3], filters=[32, 32, 128], s=2),
            IndentityBlock(512, kernels=[5, 1, 5, 3], filters=[32, 32, 128]),
            IndentityBlock(512, kernels=[5, 1, 5, 3], filters=[32, 32, 128]),
            IndentityBlock(512, kernels=[5, 1, 5, 3], filters=[32, 32, 128]),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(512, kernels=[5, 1, 5, 3], filters=[64, 64, 256], s=2),
            IndentityBlock(1024, kernels=[5, 1, 5, 3], filters=[64, 64, 256]),
            IndentityBlock(1024, kernels=[5, 1, 5, 3], filters=[64, 64, 256]),
            IndentityBlock(1024, kernels=[5, 1, 5, 3], filters=[64, 64, 256]),
            IndentityBlock(1024, kernels=[5, 1, 5, 3], filters=[64, 64, 256]),
            IndentityBlock(1024, kernels=[5, 1, 5, 3], filters=[64, 64, 256]),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(1024, kernels=[5, 1, 5, 3], filters=[128, 128, 512], s=2),
            IndentityBlock(2048, kernels=[5, 1, 5, 3], filters=[128, 128, 512]),
            IndentityBlock(2048, kernels=[5, 1, 5, 3], filters=[128, 128, 512]),
        )
        self.pool = nn.AvgPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(8192, n_class)
        )

    def forward(self, X):
        out = self.stage1(X)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.pool(out)
        out = self.pool(out)
        out = self.pool(out)
        out = out.view(out.size(0), 8192)
        out = self.fc(out)
        return out


from torch import nn
import torch.nn.functional as F
from MfA_Conv import MfA_Conv

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            MfA_Conv(in_channel, out_channel, high_L=5, high_S=1, low_L=5, low_S=3,
                     high_stride=stride, low_stride=stride, basic_stride=stride),
            nn.BatchNorm2d(out_channel*4),
            nn.ReLU(True),
            MfA_Conv(out_channel*4, out_channel, high_L=5, high_S=1, low_L=5, low_S=3),
            nn.BatchNorm2d(out_channel*4)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel*4:
            self.shortcut = nn.Sequential(
                MfA_Conv(in_channel, out_channel, high_stride=stride, low_stride=stride, basic_stride=stride),
                nn.BatchNorm2d(out_channel*4)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out



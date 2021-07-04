from torch import nn
import torch.nn.functional as F
from MfA_Conv import MfA_Conv

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, binary=True):
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

class BasicBlock_cifar(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, binary=True):
        super(BasicBlock_cifar, self).__init__()
        self.conv = MfA_Conv(inplanes, planes, high_L=5, high_S=1, low_L=5, low_S=3,
                            high_stride=stride, low_stride=stride, basic_stride=stride)
        self.bn = nn.BatchNorm2d(planes * 4)
        self.prelu = nn.PReLU(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out

class ResNet_cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, M=1, method='MConv'):
        self.inplanes = 64
        super(ResNet_cifar, self).__init__()
        self.conv1 = MfA_Conv(3, 16, high_L=5, high_S=1, low_L=5, low_S=3) # FIXME: kernel_size=3?
        self.bn1 = nn.BatchNorm2d(64 * M)
        self.prelu = nn.PReLU(64 * M)
        self.layer1 = self._make_layer(block, 16, layers[0], M=M, method=method)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, M=M, method=method)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, M=M, method=method)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, M=M, method=method)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion * M, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1, M=1, method='MConv'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion * 4:
            shortcut_conv = MfA_Conv(self.inplanes * M, planes * block.expansion)
            downsample = nn.Sequential(shortcut_conv,
                nn.BatchNorm2d(planes * block.expansion * M * 4),
                nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, binary=True))
        self.inplanes = planes * block.expansion * 4
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, binary=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.fc(x)

        return x

def resnet18_cifar(pretrained=False, M=1, method='MConv'):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_cifar(ResBlock, [2, 2, 2, 2], M=M, method=method)
    if pretrained:
        raise ValueError
    return model

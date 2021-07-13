import torch
import torch.nn as nn
import torch.nn.functional as F
import MfA_Conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MfA_Conv.MfA_Conv(in_planes, planes, high_L=3, high_S=1, low_L=3, low_S=3,
                                       high_stride=stride, low_stride=stride, basic_stride=stride)
        self.bn1 = nn.BatchNorm2d(planes * 4)
        self.conv2 = MfA_Conv.MfA_Conv(planes * 4, planes, high_L=3, high_S=1, low_L=3, low_S=3,)
        self.bn2 = nn.BatchNorm2d(planes * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes*4:
            self.shortcut = nn.Sequential(
                MfA_Conv.MfA_Conv(in_planes, planes, high_L=3, high_S=1, low_L=3, low_S=3,
                                  high_stride=stride, low_stride=stride, basic_stride=stride),
                nn.BatchNorm2d(self.expansion*planes*4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MfA_Conv.MfA_Conv(in_planes, planes, high_L=3, high_S=1, low_L=3, low_S=1)
        self.bn1 = nn.BatchNorm2d(planes * 4)
        self.conv2 = MfA_Conv.MfA_Conv(planes * 4, planes, high_L=3, high_S=1, low_L=3, low_S=3,
                                       high_stride=stride, low_stride=stride, basic_stride=stride)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = MfA_Conv.MfA_Conv(planes * 4, planes * self.expansion, high_L=3, high_S=1, low_L=3, low_S=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes*4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes*4:
            self.shortcut = nn.Sequential(
                MfA_Conv.MfA_Conv(in_planes, self.expansion*planes, high_L=1, high_S=1, low_L=1, low_S=1,
                                  high_stride=stride, low_stride=stride, basic_stride=stride),
                nn.BatchNorm2d(self.expansion*planes*4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = MfA_Conv.MfA_Conv_start(3, 4, high_L=3, high_S=1, low_L=3, low_S=3,
                                             high_stride=1, low_stride=1, basic_stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 4, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        print(out.size())
        out = self.layer2(out)
        print(out.size())
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_slim():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34_slim():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet26_slim():
    return ResNet(Bottleneck, [2, 2, 2, 2])

def ResNet50_slim():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101_slim():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152_slim():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())
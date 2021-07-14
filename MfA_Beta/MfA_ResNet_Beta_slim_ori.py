import torch
import torch.nn as nn
import torch.nn.functional as F
import MfA_Conv_Beta

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MfA_Conv_Beta.MfA_conv_start(in_planes, planes, high_L=3, high_S=1, low_L=3, low_S=3,
                                                  stride=stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = MfA_Conv_Beta.MfA_conv_end(planes*2, planes, high_L=3, high_S=1, low_L=3, low_S=3,
                                                stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.conv1 = self.conv2 = MfA_Conv_Beta.MfA_conv_start(in_planes, planes, high_L=1, high_S=1, low_L=1, low_S=1)
        self.bn1 = nn.BatchNorm2d(2 * planes)
        self.conv2 = MfA_Conv_Beta.MfA_conv(2*planes, planes, high_L=3, high_S=3, low_L=3, low_S=3,
                                            stride=stride)
        self.bn2 = nn.BatchNorm2d(2 * planes)
        self.conv3 = MfA_Conv_Beta.MfA_conv_end(planes*2, self.expansion*planes, high_L=1, high_S=1, low_L=1, low_S=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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

        self.conv1 = MfA_Conv_Beta.MfA_layer1(3, 16, high_L=3, high_S=1, low_L=3, low_S=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
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


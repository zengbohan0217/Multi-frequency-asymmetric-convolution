import torch
import torch.nn as nn
import MfA_Conv

# try:
#     from torch.hub import load_state_dict_from_url
# except ImportError:
#     from torch.utils.model_zoo import load_url as load_state_dict_from_url

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']



def conv3x3(in_planes, planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, planes, stride=1, groups=1, dilation=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module): #输出channel和输入chnnel都是64
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and basic_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = MfA_Conv.MfA_Conv(in_planes, planes, high_L=3, high_S=1, low_L=3, low_S=3,
                            high_stride=stride, low_stride=stride, basic_stride=stride)  # 若有下采样，则在第一个3x3卷积进行
        self.bn1 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MfA_Conv.MfA_Conv(planes * 4, planes, high_L=3, high_S=1, low_L=3, low_S=3)
        self.bn2 = norm_layer(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.ReLU1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)

        out = x + identity #残差结构
        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    expansion = 4 #in_planes=256, planes=64, 恢复时需要扩展为4倍
    __constants__ = ['downsample']
    def __init__(self, in_planes, planes, stride=1, downsample=None,
                    groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups #??

        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = identity + x

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            #tuple中的每个元素表示 是否用空洞卷积dilated convolution代替2x2 stride
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #224x224=>112x112  64@7x7, stride 2
        self.conv1 = MfA_Conv.MfA_Conv_start(3, 16, high_L=3, high_S=3, low_L=5, low_S=5,
                                            high_stride=2, low_stride=2, basic_stride=2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #112x112=>56x56 3x3 maxpoll, stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        prevision_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion * 4:
            downsample = nn.Sequential(
                MfA_Conv.MfA_Conv(self.inplanes, planes, high_L=1, high_S=1, low_L=1, low_S=1,
                                               high_stride=stride, low_stride=stride, basic_stride=stride),
                norm_layer(planes * block.expansion * 4)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, prevision_dilation, norm_layer))
        self.inplanes = planes * block.expansion * 4
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _foward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._foward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #     model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained, progress, **kwargs)

def resnetxt50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', BottleNeck, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', BottleNeck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', BottleNeck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', BottleNeck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
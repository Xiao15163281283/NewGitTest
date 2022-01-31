import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck1_1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck1_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 55, stride, groups, dilation)
        self.bn2 = norm_layer(55)
        self.conv3 = conv1x1(55, 168)
        self.bn3 = norm_layer(168)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck1_2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck1_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 44, stride, groups, dilation)
        self.bn2 = norm_layer(44)
        self.conv3 = conv1x1(44, 168)
        self.bn3 = norm_layer(168)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck1_3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck1_3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 42, stride, groups, dilation)
        self.bn2 = norm_layer(42)
        self.conv3 = conv1x1(42, 168)
        self.bn3 = norm_layer(168)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck2_1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 114, stride, groups, dilation)
        self.bn2 = norm_layer(114)
        self.conv3 = conv1x1(114, 472)
        self.bn3 = norm_layer(472)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck2_2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 84, stride, groups, dilation)
        self.bn2 = norm_layer(84)
        self.conv3 = conv1x1(84, 472)
        self.bn3 = norm_layer(472)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck2_3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2_3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 106, stride, groups, dilation)
        self.bn2 = norm_layer(106)
        self.conv3 = conv1x1(106, 472)
        self.bn3 = norm_layer(472)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck2_4(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2_4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 127, stride, groups, dilation)
        self.bn2 = norm_layer(127)
        self.conv3 = conv1x1(127, 472)
        self.bn3 = norm_layer(472)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck3_1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 253, stride, groups, dilation)
        self.bn2 = norm_layer(253)
        self.conv3 = conv1x1(253, 876)
        self.bn3 = norm_layer(876)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck3_2(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 241, stride, groups, dilation)
        self.bn2 = norm_layer(241)
        self.conv3 = conv1x1(241, 876)
        self.bn3 = norm_layer(876)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck3_3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3_3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 229, stride, groups, dilation)
        self.bn2 = norm_layer(229)
        self.conv3 = conv1x1(229, 876)
        self.bn3 = norm_layer(876)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck3_4(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3_4, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 254, stride, groups, dilation)
        self.bn2 = norm_layer(254)
        self.conv3 = conv1x1(254, 876)
        self.bn3 = norm_layer(876)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck3_5(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3_5, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 232, stride, groups, dilation)
        self.bn2 = norm_layer(232)
        self.conv3 = conv1x1(232, 876)
        self.bn3 = norm_layer(876)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck3_6(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck3_6, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 204, stride, groups, dilation)
        self.bn2 = norm_layer(204)
        self.conv3 = conv1x1(204, 876)
        self.bn3 = norm_layer(876)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck4_1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck4_1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 479, stride, groups, dilation)
        self.bn2 = norm_layer(479)
        self.conv3 = conv1x1(479, 1681)
        self.bn3 = norm_layer(1681)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck4_2(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck4_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 275, stride, groups, dilation)
        self.bn2 = norm_layer(275)
        self.conv3 = conv1x1(275, 1681)
        self.bn3 = norm_layer(1681)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck4_3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck4_3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        # print(self.conv1)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, 202, stride, groups, dilation)
        self.bn2 = norm_layer(202)
        self.conv3 = conv1x1(202, 1681)
        self.bn3 = norm_layer(1681)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Resnet50(nn.Module):
    def __init__(self, num_class=100):
        super(Resnet50, self).__init__()

        self.inplanes = 64

        # -----------
        # Pre-layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ------------------
        # layer1 | 3 block
        down_1 = nn.Sequential(conv1x1(self.inplanes, 168, stride=1),
                               nn.BatchNorm2d(168))
        plane = 33
        self.layer1_1 = Bottleneck1_1(self.inplanes, plane, stride=1, downsample=down_1)
        # self.inplanes = plane * Bottleneck.expansion  # 256
        self.inplanes = 168
        plane = 35
        self.layer1_2 = Bottleneck1_2(self.inplanes, plane, stride=1)
        plane = 37
        self.layer1_3 = Bottleneck1_3(self.inplanes, plane, stride=1)

        # ------------------
        down_2 = nn.Sequential(conv1x1(self.inplanes, 472, stride=2),
                               nn.BatchNorm2d(472))
        plane = 77
        self.layer2_1 = Bottleneck2_1(self.inplanes, plane, stride=2, downsample=down_2)
        self.inplanes = 472  # 512
        plane = 43
        self.layer2_2 = Bottleneck2_2(self.inplanes, plane, stride=1)
        plane = 96
        self.layer2_3 = Bottleneck2_3(self.inplanes, plane, stride=1)
        plane = 121
        self.layer2_4 = Bottleneck2_4(self.inplanes, plane, stride=1)

        # ------------------
        # layer3 | 6 block
        down_3 = nn.Sequential(conv1x1(self.inplanes, 876, stride=2),
                               nn.BatchNorm2d(876))
        plane = 240
        self.layer3_1 = Bottleneck3_1(self.inplanes, plane, stride=2, downsample=down_3)
        self.inplanes = 876  # 1024
        plane = 219
        self.layer3_2 = Bottleneck3_2(self.inplanes, plane, stride=1)
        plane = 239
        self.layer3_3 = Bottleneck3_3(self.inplanes, plane, stride=1)
        plane = 247
        self.layer3_4 = Bottleneck3_4(self.inplanes, plane, stride=1)
        plane = 241
        self.layer3_5 = Bottleneck3_5(self.inplanes, plane, stride=1)
        plane = 207
        self.layer3_6 = Bottleneck3_6(self.inplanes, plane, stride=1)

        # ------------------
        # layer4 | 3 block
        down_4 = nn.Sequential(conv1x1(self.inplanes, 1681, stride=2),
                               nn.BatchNorm2d(1681))
        plane = 511
        self.layer4_1 = Bottleneck4_1(self.inplanes, plane, stride=2, downsample=down_4)
        self.inplanes = 1681  # 2048
        plane = 187
        self.layer4_2 = Bottleneck4_2(self.inplanes, plane, stride=1)
        plane = 131
        self.layer4_3 = Bottleneck4_3(self.inplanes, plane, stride=1)

        # ----
        # GAP
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        # ----------------
        # Classification
        self.classifier = nn.Linear(1681, num_class)
    def forward(self, x):
        # [Pre-layer] | inputshape [batch_size, 3, 256, 128] |
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print('x.shape',x.shape)

        # [layer1] | inputshape [batch_size, 64, 64, 32]
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = self.layer1_3(x)
        # print('x1.shape', x.shape)
        # [layer2] | inputshape [batch_size, 64, 64, 32]
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)
        x = self.layer2_4(x)
        # print('x2.shape', x.shape)
        # [layer3] | inputshape [batch_size, 512, 32, 16]
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)
        x = self.layer3_5(x)
        x = self.layer3_6(x)
        # print('x3.shape', x.shape)
        # [layer4] inputshape [batch_size, 1024, 16, 8]
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)
        # print('x4.shape', x.shape)
        # GAP
        x = self.GAP(x)
        # print('self.GAP',self.GAP)
        # print('============================')
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1])
        # print(x.shape,)
        # Classification
        x = self.classifier(x)

        return x


# if __name__ == "__main__":
#     import torch
#     #
#     input = torch.FloatTensor(torch.rand([1,3,32,32]))
#     net = Resnet50()
#     # print('net',net)
#     output = net(input)
#     # print('', net)

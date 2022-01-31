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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
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
    def __init__(self, num_class=10):
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
        plane = 64
        down_1 = nn.Sequential(conv1x1(self.inplanes, plane * Bottleneck.expansion, stride=1),
                               nn.BatchNorm2d(plane * Bottleneck.expansion))

        self.layer1_1 = Bottleneck(self.inplanes, plane, stride=1, downsample=down_1)
        self.inplanes = plane * Bottleneck.expansion  # 256
        self.layer1_2 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer1_3 = Bottleneck(self.inplanes, plane, stride=1)

        # ------------------
        # layer2 | 4 block
        plane = 128
        down_2 = nn.Sequential(conv1x1(self.inplanes, plane * Bottleneck.expansion, stride=2),
                               nn.BatchNorm2d(plane * Bottleneck.expansion))
        self.layer2_1 = Bottleneck(self.inplanes, plane, stride=2, downsample=down_2)
        self.inplanes = plane * Bottleneck.expansion  # 512
        self.layer2_2 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer2_3 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer2_4 = Bottleneck(self.inplanes, plane, stride=1)

        # ------------------
        # layer3 | 6 block
        plane = 256
        down_3 = nn.Sequential(conv1x1(self.inplanes, plane * Bottleneck.expansion, stride=2),
                               nn.BatchNorm2d(plane * Bottleneck.expansion))
        self.layer3_1 = Bottleneck(self.inplanes, plane, stride=2, downsample=down_3)
        self.inplanes = plane * Bottleneck.expansion  # 1024
        self.layer3_2 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer3_3 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer3_4 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer3_5 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer3_6 = Bottleneck(self.inplanes, plane, stride=1)

        # ------------------
        # layer4 | 3 block
        plane = 512
        down_4 = nn.Sequential(conv1x1(self.inplanes, plane * Bottleneck.expansion, stride=2),
                               nn.BatchNorm2d(plane * Bottleneck.expansion))
        self.layer4_1 = Bottleneck(self.inplanes, plane, stride=2, downsample=down_4)
        self.inplanes = plane * Bottleneck.expansion  # 2048
        self.layer4_2 = Bottleneck(self.inplanes, plane, stride=1)
        self.layer4_3 = Bottleneck(self.inplanes, plane, stride=1)

        # ----
        # GAP
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        # ----------------
        # Classification
        self.classifier = nn.Linear(2048, num_class)

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
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1])

        # Classification
        x = self.classifier(x)

        return x


# if __name__ == "__main__":
#     import torch
#     #
#     input = torch.FloatTensor(torch.rand([1,3,32,32]))
#     net = Resnet50()
#     output = net(input)
#     # print('', net)

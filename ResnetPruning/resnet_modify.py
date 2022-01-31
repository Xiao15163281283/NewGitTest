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
        # print(self.conv1)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    #
    # def forward(self, x):
    #     identity = x
    #
    #     out = self.conv1(x)
    #
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #
    #     out = self.conv2(out)
    #     out = self.bn2(out)
    #     out = self.relu(out)
    #
    #     out = self.conv3(out)
    #     out = self.bn3(out)
    #
    #     if self.downsample is not None:
    #         identity = self.downsample(x)
    #
    #     out += identity
    #     out = self.relu(out)
    #
    #     return out


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
        # print(x.shape)
        """[layer1] | inputshape [batch_size, 64, 64, 32]"""
        x0 = x
        x = self.layer1_1.conv1(x)
        layer1_1_1 = x
        x = self.layer1_1.bn1(x)
        x = self.layer1_1.conv2(x)
        layer1_1_2 = x
        x = self.layer1_1.bn2(x)
        x = self.layer1_1.conv3(x)
        x = self.layer1_1.bn3(x)
        x0 = self.layer1_1.downsample(x0)
        x += x0
        # x=self.layer1_1(x)
        # layer1_1_3 = x
        x = self.layer1_1.relu(x)
        # x = self.layer1_2(x)
        x0 = x
        x = self.layer1_2.conv1(x)
        layer1_2_1 = x
        x = self.layer1_2.bn1(x)
        x = self.layer1_2.conv2(x)
        layer1_2_2 = x
        x = self.layer1_2.bn2(x)
        x = self.layer1_2.conv3(x)
        x = self.layer1_2.bn3(x)
        x += x0
        # layer1_2_3 = x
        x = self.layer1_2.relu(x)
        # print(self.layer1_3)
        x0 = x
        x = self.layer1_3.conv1(x)
        layer1_3_1 = x
        x = self.layer1_3.bn1(x)
        x = self.layer1_3.conv2(x)
        layer1_3_2 = x
        x = self.layer1_3.bn2(x)
        x = self.layer1_3.conv3(x)
        x = self.layer1_3.bn3(x)
        x += x0
        layer1_3_3 = x
        x = self.layer1_3.relu(x)
        # print('layer1_3.shape',x.shape)
        """[layer2] | inputshape [batch_size, 64, 64, 32]"""
        # x = self.layer2_1(x)
        x0 = x
        x = self.layer2_1.conv1(x)
        layer2_1_1 = x
        x = self.layer2_1.bn1(x)
        x = self.layer2_1.conv2(x)
        layer2_1_2 = x
        x = self.layer2_1.bn2(x)
        x = self.layer2_1.conv3(x)
        # layer2_1_3 = x
        x = self.layer2_1.bn3(x)
        x0 = self.layer2_1.downsample(x0)
        x += x0
        # layer2_1_4 = x
        x = self.layer2_1.relu(x)
        # x = self.layer2_2(x)
        x0 = x
        x = self.layer2_2.conv1(x)
        layer2_2_1 = x
        x = self.layer2_2.bn1(x)
        x = self.layer2_2.conv2(x)
        layer2_2_2 = x
        x = self.layer2_2.bn2(x)
        x = self.layer2_2.conv3(x)
        # layer2_2_3 = x
        x = self.layer2_2.bn3(x)
        x += x0
        # layer2_2_4 = x
        x = self.layer2_2.relu(x)
        # x = self.layer2_3(x)
        x0 = x
        x = self.layer2_3.conv1(x)
        layer2_3_1 = x
        x = self.layer2_3.bn1(x)
        x = self.layer2_3.conv2(x)
        layer2_3_2 = x
        x = self.layer2_3.bn2(x)
        x = self.layer2_3.conv3(x)
        # layer2_3_3 = x
        x = self.layer2_3.bn3(x)
        x += x0
        # layer2_3_4 = x
        x = self.layer2_3.relu(x)
        # x = self.layer2_4(x)
        x0 = x
        x = self.layer2_4.conv1(x)
        layer2_4_1 = x
        x = self.layer2_4.bn1(x)
        x = self.layer2_4.conv2(x)
        layer2_4_2 = x
        x = self.layer2_4.bn2(x)
        x = self.layer2_4.conv3(x)
        # layer2_4_3 = x
        x = self.layer2_4.bn3(x)
        x += x0
        layer2_4_3 = x
        x = self.layer2_4.relu(x)
        # [layer3] | inputshape [batch_size, 512, 32, 16]
        # x = self.layer3_1(x)
        x0 = x
        x = self.layer3_1.conv1(x)
        layer3_1_1 = x
        x = self.layer3_1.bn1(x)
        x = self.layer3_1.conv2(x)
        layer3_1_2 = x
        x = self.layer3_1.bn2(x)
        x = self.layer3_1.conv3(x)
        # layer3_1_3 = x
        x = self.layer3_1.bn3(x)
        x0 = self.layer3_1.downsample(x0)
        x += x0
        # layer3_1_4 = x
        x = self.layer3_1.relu(x)
        # x = self.layer3_2(x)
        x0 = x
        x = self.layer3_2.conv1(x)
        layer3_2_1 = x
        x = self.layer3_2.bn1(x)
        x = self.layer3_2.conv2(x)
        layer3_2_2 = x
        x = self.layer3_2.bn2(x)
        x = self.layer3_2.conv3(x)
        # layer3_2_3 = x
        x = self.layer3_2.bn3(x)
        x += x0
        # layer3_2_4 = x
        x = self.layer3_2.relu(x)
        # x = self.layer3_3(x)
        x0 = x
        x = self.layer3_3.conv1(x)
        layer3_3_1 = x
        x = self.layer3_3.bn1(x)
        x = self.layer3_3.conv2(x)
        layer3_3_2 = x
        x = self.layer3_3.bn2(x)
        x = self.layer3_3.conv3(x)
        # layer3_3_3 = x
        x = self.layer3_3.bn3(x)
        x += x0
        # layer3_3_4 = x
        x = self.layer3_3.relu(x)
        # x = self.layer3_4(x)
        x0 = x
        x = self.layer3_4.conv1(x)
        layer3_4_1 = x
        x = self.layer3_4.bn1(x)
        x = self.layer3_4.conv2(x)
        layer3_4_2 = x
        x = self.layer3_4.bn2(x)
        x = self.layer3_4.conv3(x)
        # layer3_4_3 = x
        x = self.layer3_4.bn3(x)
        x += x0
        # layer3_4_4 = x
        x = self.layer3_4.relu(x)
        # x = self.layer3_5(x)
        x0 = x
        x = self.layer3_5.conv1(x)
        layer3_5_1 = x
        x = self.layer3_5.bn1(x)
        x = self.layer3_5.conv2(x)
        layer3_5_2 = x
        x = self.layer3_5.bn2(x)
        x = self.layer3_5.conv3(x)
        # layer3_5_3 = x
        x = self.layer3_5.bn3(x)
        x += x0
        # layer3_5_4 = x
        x = self.layer3_5.relu(x)
        # x = self.layer3_6(x)
        x0 = x
        x = self.layer3_6.conv1(x)
        layer3_6_1 = x
        x = self.layer3_6.bn1(x)
        x = self.layer3_6.conv2(x)
        layer3_6_2 = x
        x = self.layer3_6.bn2(x)
        x = self.layer3_6.conv3(x)
        # layer3_6_3 = x
        x = self.layer3_6.bn3(x)
        x += x0
        layer3_6_3 = x
        x = self.layer3_6.relu(x)
        # print('x3.shape', x.shape)
        # [layer4] inputshape [batch_size, 1024, 16, 8]
        # x = self.layer4_1(x)
        x0 = x
        x = self.layer4_1.conv1(x)
        layer4_1_1 = x
        x = self.layer4_1.bn1(x)
        x = self.layer4_1.conv2(x)
        layer4_1_2 = x
        x = self.layer4_1.bn2(x)
        x = self.layer4_1.conv3(x)
        # layer4_1_3 = x
        x = self.layer4_1.bn3(x)
        x0 = self.layer4_1.downsample(x0)
        x += x0
        # layer4_1_4 = x
        x = self.layer4_1.relu(x)
        # x = self.layer4_2(x)
        x0 = x
        x = self.layer4_2.conv1(x)
        layer4_2_1 = x
        x = self.layer4_2.bn1(x)
        x = self.layer4_2.conv2(x)
        layer4_2_2 = x
        x = self.layer4_2.bn2(x)
        x = self.layer4_2.conv3(x)
        # layer4_2_3 = x
        x = self.layer4_2.bn3(x)
        x += x0
        # layer4_2_4 = x
        x = self.layer4_2.relu(x)
        # x = self.layer4_3(x)
        x0 = x
        x = self.layer4_3.conv1(x)
        layer4_3_1 = x
        x = self.layer4_3.bn1(x)
        x = self.layer4_3.conv2(x)
        layer4_3_2 = x
        x = self.layer4_3.bn2(x)
        x = self.layer4_3.conv3(x)
        # layer4_3_3 = x
        x = self.layer4_3.bn3(x)
        x += x0
        layer4_3_3 = x
        x = self.layer4_3.relu(x)
        # print('x4.shape', x.shape)
        # GAP
        x = self.GAP(x)
        # print(x.shape)
        x = x.view(x.shape[0], x.shape[1])

        # Classification
        x = self.classifier(x)

        return x,layer1_1_1,layer1_1_2,layer1_2_1,layer1_2_2,layer1_3_1,layer1_3_2,layer1_3_3,layer2_1_1,layer2_1_2,layer2_2_1,layer2_2_2,layer2_3_1,layer2_3_2,layer2_4_1,layer2_4_2,layer2_4_3,layer3_1_1,layer3_1_2,layer3_2_1,layer3_2_2,layer3_3_1,layer3_3_2,layer3_4_1,layer3_4_2,layer3_5_1,layer3_5_2,layer3_6_1,layer3_6_2,layer3_6_3,layer4_1_1,layer4_1_2,layer4_2_1,layer4_2_2,layer4_3_1,layer4_3_2,layer4_3_3,


# if __name__ == "__main__":
#     import torch
#     input = torch.FloatTensor(torch.rand([1,3,32,32]))
#     net = Resnet50()
#     x,layer1_1_1,layer1_1_2,layer1_2_1,layer1_2_2,layer1_3_1,layer1_3_2,layer1_3_3,layer2_1_1,layer2_1_2,layer2_2_1,layer2_2_2,layer2_3_1,layer2_3_2,layer2_4_1,layer2_4_2,layer2_4_3,layer3_1_1,layer3_1_2,layer3_2_1,layer3_2_2,layer3_3_1,layer3_3_2,layer3_4_1,layer3_4_2,layer3_5_1,layer3_5_2,layer3_6_1,layer3_6_2,layer3_6_3,layer4_1_1,layer4_1_2,layer4_2_1,layer4_2_2,layer4_3_1,layer4_3_2,layer4_3_3 = net(input)
#     print('layer1_3_3', layer1_3_3.shape)

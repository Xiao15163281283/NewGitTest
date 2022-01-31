from torch import nn
import torch

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch))
    # Make sure that round down does not go down by more than 10%.
    # if new_ch < 0.9 * ch:
    #     new_ch += divisor
    return new_ch

# groups=1默认的是普通卷积，groups=in_channel默认是DW卷积
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

#倒残差结构  expand_ratio：扩展因子
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual1(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual1, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual4(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual4, self).__init__()
        hidden_channel = 373
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual5(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual5, self).__init__()
        hidden_channel = 556
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)
class InvertedResidual6(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual6, self).__init__()
        hidden_channel = 910
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        block4 = InvertedResidual4
        block5 = InvertedResidual5
        block6 = InvertedResidual6
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(535 * alpha, round_nearest)
        # t, c, n, s
        inverted_residual_setting1 = [[1, 16, 1, 1], ]
        inverted_residual_setting2 = [[6, 24, 2, 2], ]
        inverted_residual_setting3 = [[6, 32, 3, 2], ]
        inverted_residual_setting4 = [[6, 53, 4, 2], ]
        inverted_residual_setting5 = [[6, 76, 3, 1], ]
        inverted_residual_setting6 = [[6, 110, 3, 2], ]
        inverted_residual_setting7 = [[6, 136, 1, 1], ]
        #
        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=1))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting1:
            # print('t,c, n, s', t,c,n,s)
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting2:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting3:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting4:
            output_channel = _make_divisible(c * alpha, round_nearest)
            features.append(block(input_channel, output_channel, 2, expand_ratio=t))
            input_channel = output_channel
            for i in range(n-1):
                stride = s if i == 0 else 1
                features.append(block4(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting5:
            output_channel = _make_divisible(c * alpha, round_nearest)
            # print('output_channel', output_channel)
            # print('input_channel', input_channel)
            features.append(block4(input_channel, output_channel, stride, expand_ratio=t))
        #     print(block)
            input_channel = output_channel
            for i in range(n-1):
                stride = s if i == 0 else 1
                features.append(block5(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting6:
            output_channel = _make_divisible(c * alpha, round_nearest)
            # print('output_channel', output_channel)
            # print('input_channel', input_channel)
            features.append(block5(input_channel, output_channel, 2, expand_ratio=t))
        # #     print(block)
            input_channel = output_channel
            for i in range(n-1):
                stride = s if i == 0 else 1
                features.append(block6(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        for t, c, n, s in inverted_residual_setting7:
            output_channel = _make_divisible(c * alpha, round_nearest)
            features.append(block6(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
            for i in range(n-1):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers

        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.xavier_normal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MobileNetV2()
    print(model)
    # print(model.features[15])
    # print(model.features[16])
    # print(model.features[17])
    # # print(model)
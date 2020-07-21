import torch
import torch.nn as nn
import logging
import math

from .quant import conv3x3, conv1x1
from .layers import norm

def seq_c_b_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_b_a_c_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = relu(out)
    out = conv(out)
    if skip_enable:
        out += skip
    return out

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, args):
        super(InvertedResidual, self).__init__()
        self.args = args
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio

        self.use_res_connect = self.stride == 1 and inp == oup

        if 'cbas' in args.keyword: # default ?
            self.seq = seq_c_b_a_s
        elif 'bacs' in args.keyword:
            self.seq = seq_b_a_c_s
        else:
            self.seq = None

        setattr(self, 'relu1', nn.ReLU(inplace=True))
        if 'cbas' in args.keyword:
            if expand_ratio == 1:
                setattr(self, 'relu2', nn.Sequential())
            else:
                setattr(self, 'relu2', nn.ReLU(inplace=True))
            setattr(self, 'relu3', nn.Sequential())
        else:
            setattr(self, 'relu2', nn.ReLU(inplace=True))
            if expand_ratio == 1:
                setattr(self, 'relu3', nn.Sequential())
            else:
                setattr(self, 'relu3', nn.ReLU(inplace=True))

        if expand_ratio == 1:
            self.conv1 = conv3x3(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, args=args)
            self.conv2 = conv1x1(hidden_dim, oup, stride=1, args=args)
            self.conv3 = nn.Sequential()

            self.bn1 = norm(hidden_dim, args)
            self.bn3 = nn.Sequential()
            if 'cbas' in args.keyword:
                self.bn2 = norm(oup, args)
            elif 'bacs' in args.keyword:
                self.bn2 = norm(hidden_dim, args)
        else:
            self.conv1 = conv1x1(inp, hidden_dim, stride=1, args=args)
            self.conv2 = conv3x3(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, args=args)
            self.conv3 = conv1x1(hidden_dim, oup, stride=1, args=args)

            self.bn2 = norm(hidden_dim, args)
            if 'cbas' in args.keyword:
                self.bn1 = norm(hidden_dim, args)
                self.bn3 = norm(oup, args)
            elif 'bacs' in args.keyword:
                self.bn1 = norm(inp, args)
                self.bn3 = norm(hidden_dim, args)

        if 'bacs' in args.keyword and self.use_res_connect: # additional BN ?
            self.skip_bn = norm(oup, args)
            self.stem_bn = norm(oup, args)

    def forward(self, x):
        out = self.seq(  x, self.conv1, self.relu1, self.bn1)
        out = self.seq(out, self.conv2, self.relu2, self.bn2)
        out = self.seq(out, self.conv3, self.relu3, self.bn3)

        if self.use_res_connect:
            if 'bacs' in self.args.keyword:
                result = self.skip_bn(x) + self.stem_bn(out)
            else:
                result = x + out
        else:
            result = out
        return result


class MobileNetV2(nn.Module):
    def __init__(self, args):
        super(MobileNetV2, self).__init__()
        self.args = args
        self.width_alpha = getattr(args, 'width_alpha', 1.0)

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert args.input_size % 32 == 0
        input_channel = int(input_channel * args.width_alpha)
        self.last_channel = int(last_channel * args.width_alpha) if args.width_alpha > 1.0 else last_channel

        if 'cifar10' in args.keyword or 'cifar100' in args.keyword:
            interverted_residual_setting[1][3] = 1
            # or
            #interverted_residual_setting[5][3] = 1
            first_stride = 1
        else:
            first_stride = 2

        if 'relu6' in args.keyword:
            nn.ReLU = nn.ReLU6

        if 'preBN' in args.keyword:
            self.features = [nn.Conv2d(3, input_channel, 3, first_stride, 1, bias=False)]
        else:
            self.features = [conv_bn(3, input_channel, first_stride)]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * args.width_alpha)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, t, args))
                else:
                    self.features.append(block(input_channel, output_channel, 1, t, args))
                input_channel = output_channel

        # building last several layers
        self.features.append(nn.Sequential(
                conv1x1(input_channel, self.last_channel, stride=1, args=args),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU(inplace=True)
                ))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(self.last_channel, args.num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2) #  mean vs avg_pooling
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class conv_dw(nn.Module):
    def __init__(self, inp, outp, stride, args):
        super(conv_dw, self).__init__()
        self.stride = stride
        self.outp = outp
        self.inp = inp
        self.args = args
        self.base = getattr(args, 'base', 1)
        assert self.base == 1, 'base > 1 not supported'

        if self.base == 1:
            self.scale1 = [1]
            self.scale2 = [1]
        else:
            self.scale1 = nn.ParameterList([nn.Parameter(torch.ones(1) / self.base, requires_grad=True) for j in range(self.base)])
            self.scale2 = nn.ParameterList([nn.Parameter(torch.ones(1) / self.base, requires_grad=True) for j in range(self.base)])

        for i in range(2):
            setattr(self, 'relu%d' % (i+1), nn.ModuleList([nn.ReLU(inplace=True) for j in range(self.base)]))

        if 'cbas' in args.keyword: # default ?
            self.seq = seq_c_b_a_s
        elif 'bacs' in args.keyword:
            self.seq = seq_b_a_c_s
        else:
            self.seq = None

        keep_depth_conv = 'keep_dp' in args.keyword
        keep_point_conv = 'keep_pt' in args.keyword
        self.depth_conv = nn.ModuleList([conv3x3(inp, inp, stride=stride, groups=inp, args=args, force_fp=keep_depth_conv) for i in range(self.base)])
        self.point_conv = nn.ModuleList([conv1x1(inp, outp, stride=1, args=args, force_fp=keep_point_conv) for i in range(self.base)])

        self.bn1 = nn.ModuleList([norm(inp, args) for j in range(self.base)])
        if 'cbas' in args.keyword:
            self.bn2 = nn.ModuleList([norm(outp, args) for j in range(self.base)])
        elif 'bacs' in args.keyword:
            self.bn2 = nn.ModuleList([norm(inp, args) for j in range(self.base)])

        self.point_scale = nn.ModuleList([None for j in range(self.base)])
        if 'origin' not in args.keyword:
            if 'depth_skip' in args.keyword:
                if 'avg_pool' in args.keyword:
                    self.depth_skip = nn.Sequential(nn.AvgPool2d(stride))
                else:
                    self.depth_skip = nn.Sequential(nn.MaxPool2d(stride))

            if 'point_scale' in args.keyword:
                self.point_scale = nn.ParameterList([nn.Parameter(torch.ones(inp, 1, 1), requires_grad=True) for j in range(self.base)])

            if 'point_skip' in args.keyword:
                self.point_skip = nn.Sequential(
                        conv1x1(inp, outp, stride=1, args=args),
                        nn.BatchNorm2d(outp),
                        nn.ReLU(inplace=True)
                        )

    def forward(self, x):
        result = None
        if 'origin' not in self.args.keyword and 'depth_skip' in self.args.keyword:
            result = self.depth_skip(x)

        for depth, bn, relu, scale in zip(self.depth_conv, self.bn1, self.relu1, self.scale1):
            out = self.seq(x, depth, relu, bn)
            if result is None:
                result = scale * out
            else:
                result = result + scale * out

        #if 'origin' not in self.args.keyword and 'point_skip' in self.args.keyword:
        #    result = self.point_skip(output)

        output = result
        result = None
        for point_scale, point, bn, relu, scale in zip(self.point_scale, self.point_conv, self.bn2, self.relu2, self.scale2):
            if point_scale is not None:
                output = output * point_scale
            out = self.seq(output, point, relu, bn)
            if result is None:
                result = scale * out
            else:
                result = result + scale * out
        return result

class MobileNetV1(nn.Module):
    def __init__(self, args):
        super(MobileNetV1, self).__init__()
        self.args = args
        # width_alpha enable after 2019.12.11
        self.width_alpha = getattr(args, 'width_alpha', 1.0)
        self.inplanes = int(32 * self.width_alpha)

        if 'cifar10' in args.keyword or 'cifar100' in args.keyword:
            first_stride = 1
            downsample_size = 16
        else:
            first_stride = 2
            downsample_size = 32

        if 'preBN' in args.keyword:
            self.root = nn.Conv2d(3, self.inplanes, 3, first_stride, 1, bias=False)
            #self.pooling = nn.AvgPool2d(args.input_size // downsample_size)
            self.pooling = nn.Sequential(nn.BatchNorm2d(1024), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1)))
        else:
            self.root = conv_bn(3, self.inplanes, first_stride)
            #self.pooling = nn.AvgPool2d(args.input_size // downsample_size)
            self.pooling = nn.AdaptiveAvgPool2d((1,1))

        bottle = conv_dw
        self.model = nn.Sequential(
            bottle(int(self.width_alpha *  32), int(self.width_alpha *  64), 1, args),
            bottle(int(self.width_alpha *  64), int(self.width_alpha * 128), 2, args),
            bottle(int(self.width_alpha * 128), int(self.width_alpha * 128), 1, args),
            bottle(int(self.width_alpha * 128), int(self.width_alpha * 256), 2, args),
            bottle(int(self.width_alpha * 256), int(self.width_alpha * 256), 1, args),
            bottle(int(self.width_alpha * 256), int(self.width_alpha * 512), 2, args),
            bottle(int(self.width_alpha * 512), int(self.width_alpha * 512), 1, args),
            bottle(int(self.width_alpha * 512), int(self.width_alpha * 512), 1, args),
            bottle(int(self.width_alpha * 512), int(self.width_alpha * 512), 1, args),
            bottle(int(self.width_alpha * 512), int(self.width_alpha * 512), 1, args),
            bottle(int(self.width_alpha * 512), int(self.width_alpha * 512), 1, args),
            bottle(int(self.width_alpha * 512), int(self.width_alpha *1024), 2, args),
            bottle(int(self.width_alpha *1024), int(self.width_alpha *1024), 1, args),
        )

        self.classifier = nn.Sequential (
            #nn.Dropout(0.5),
            nn.Linear(int(self.width_alpha * 1024), args.num_classes)
        )

    def forward(self, x):
        x = self.root(x)
        x = self.model(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenetv2(args):
    model = MobileNetV2(args)
    return model

def mobilenetv1(args):
    model = MobileNetV1(args)
    return model




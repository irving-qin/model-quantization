
import torch
import torch.nn as nn
from copy import deepcopy

from .quant import conv3x3, conv1x1, conv0x0

def seq_c_b_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_c_b_s_a(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    if skip_enable:
        out += skip
    out = relu(out)
    return out

def seq_c_a_b_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = relu(out)
    out = bn(out)
    if skip_enable:
        out += skip
    return out

def seq_b_c_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = conv(out)
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

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class StaticBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

def norm(channel, args=None, feature_stride=None):
    keyword = None
    if args is not None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        return nn.BatchNorm2d(channel)

    if "group-norm" in keyword:
        group = getattr(args, "fm_quant_group", 2)
        return nn.GroupNorm(group, channel)

    if "static-bn" in keyword:
        return StaticBatchNorm2d(channel)

    if "freeze-bn" in keyword:
        return FrozenBatchNorm2d(channel)

    return nn.BatchNorm2d(channel)

class ShiftReLU(nn.ReLU):
    def __init__(self, args):
        super(ShiftReLU, self).__init__(inplace=True)
        self.shift = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        x = x + self.shift
        x = super(ShiftReLU, self).forward(x)
        return x


def actv(args=None):
    keyword = None
    if args is not None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        return nn.ReLU(inplace=True)

    if 'PReLU' in keyword:
        return nn.PReLU()

    if 'NReLU' in keyword:
        return nn.Sequential()

    if 'SReLU' in keyword:
        return ShiftReLU(args)

    return nn.ReLU(inplace=True)

# TResNet: High Performance GPU-Dedicated Architecture (https://arxiv.org/pdf/2003.13630v1.pdf)
class TResNetStem(nn.Module):
    def __init__(self, out_channel, in_channel=3, stride=4, kernel_size=1, args=None):
        super(TResNetStem, self).__init__()
        self.stride = stride
        force_fp = True
        if hasattr(args, 'keyword'):
            force_fp = 'real_skip' in args.keyword
        assert kernel_size in [1, 3], "Error reshape conv kernel"
        if kernel_size == 1:
            self.conv = conv1x1(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)
        elif kernel_size == 3:
            self.conv = conv3x3(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.stride, self.stride, W // self.stride, self.stride)
        x = x.transpose(4, 3).reshape(B, C, 1, H // self.stride, W // self.stride, self.stride * self.stride)
        x = x.transpose(2, 5).reshape(B, C * self.stride * self.stride, H // self.stride, W // self.stride)
        x = self.conv(x)
        return x

class DuplicateModule(nn.Module):
    def __init__(self, module, num):
        super(DuplicateModule, self).__init__()
        assert num >=1, "Num should greater or equal 1"

        self.model = module
        self.duplicates = []
        for i in range(1, num):
            self.duplicates.append(deepcopy(self.module))

    def forward(self, x):
        result = []
        result.append(self.model(x))
        for model in self.duplicates:
            result.append(model(x))

        if len(result) > 1:
            return torch.cat(result, dim=1)
        else:
            return result[0]

def Duplicate(module, num=1):
    return DuplicateModule(module, num)



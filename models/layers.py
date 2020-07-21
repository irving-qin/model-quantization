
import torch
import torch.nn as nn

from .quant import conv3x3, conv1x1, conv0x0

class SliceBN(nn.Module):
    def __init__(self, channel, group):
        super(SliceBN, self).__init__()
        self.group = group
        self.bn = nn.ModuleList([nn.BatchNorm2d(channel) for i in range(group * group)])

    def forward(self, x):
        group = self.group
        if group == 1:
            return self.bn[0](x)
        else:
            shape = x.shape
            x = x.reshape(shape[0], shape[1], group, shape[2] // group, group, shape[3] // group)
            y = x.new_zeros(x.shape)
            for i in range(self.group):
                for j in range(self.group):
                    slices = x[:,:, i, :, j, :]
                    slices = self.bn[i*group+j](slices)
                    y[:,:, i, :, j, :] = slices
            y = y.reshape(shape)
            return y

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

    if 'spatial' in keyword:
        quant_group = getattr(args, 'fm_quant_group', None)
        if quant_group is None or feature_stride is None:
            return nn.BatchNorm2d(channel)

        quant_group = quant_group // feature_stride
        quant_group = 1 if quant_group < 1 else quant_group
        return SliceBN(channel, group=quant_group)

    if "static-bn" in keyword:
        return StaticBatchNorm2d(channel)

    if "freeze-bn" in keyword:
        return FrozenBatchNorm2d(channel)

    return nn.BatchNorm2d(channel)

class EnchanceReLU(nn.ReLU):
    def __init__(self, args):
        super(EnchanceReLU, self).__init__(inplace=True)
        self.shift = getattr(args, 'fm_boundary', 0.25)

    def forward(self, x):
        x = x + self.shift
        x = super(EnchanceReLU, self).forward(x)
        x = x - self.shift
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

    if 'enhance-info' in keyword:
        return EnchanceReLU(args)

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


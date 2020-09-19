import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x, clip_value):
    # x = x / clip_value
    # x = torch.where(x.abs() < 1, x, x.sign())

    x = x / clip_value
    x = torch.clamp(x, min=-1, max=1)
    return x


def normalization_on_activations(x, clip_value):
    # x = F.relu(x)
    # x = x / clip_value
    # x = torch.where(x < 1, x, x.new_ones(x.shape))

    x = x / clip_value
    x = torch.clamp(x, min=0, max=1)
    return x


def quantize_activation(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k)
    x = x * clip_value
    return x


def quantize_weight(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0
    x = quantization(x, k)
    x = x * 2.0 - 1.0
    x = x * clip_value
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bits_weights=32,
        bits_activations=32,
    ):
        super(QConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        # self.eps = 1e-5
        self.weight_clip_value = nn.Parameter(torch.Tensor([1]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1]))
        self.bits_weights = int(bits_weights)
        self.bits_activations = int(bits_activations)

    def forward(self, input):
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_clip_value.abs()
        )
        weight_mean = self.weight.data.mean()
        weight_std = self.weight.data.std()
        normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        quantized_weight = quantize_weight(
            normalized_weight, self.bits_weights, self.weight_clip_value.abs()
        )
        output = F.conv2d(
            quantized_input,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        self.output_shape = output.shape
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_clip_rcf_wn_conv")
        return s


class QLinear(nn.Linear):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self, in_features, out_features, bias=True, bits_weights=32, bits_activations=32
    ):
        super(QLinear, self).__init__(in_features, out_features, bias=bias)
        # self.eps = 1e-5
        self.init_state = False
        self.weight_clip_value = nn.Parameter(torch.Tensor([1]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1]))
        self.bits_weights = int(bits_weights)
        self.bits_activations = int(bits_activations)
        # scale_factor = out_features * [0.01]
        # self.scale_factor = nn.Parameter(torch.FloatTensor(scale_factor))

    def forward(self, input):
        quantized_input = quantize_activation(
            input, self.bits_activations, self.activation_clip_value.abs()
        )
        # weight_mean = self.weight.data.mean()
        # weight_std = self.weight.data.std()
        # normalized_weight = self.weight.add(-weight_mean).div(weight_std)
        # quantized_weight = quantize_weight(normalized_weight, self.bits_weights, self.weight_clip_value.abs())
        quantized_weight = quantize_weight(self.weight, self.bits_weights, self.weight_clip_value.abs())
        output = F.linear(quantized_input, quantized_weight, self.bias)
        # output = output * self.scale_factor
        self.output_shape = output.shape
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_rcf_wn_linear")
        return s

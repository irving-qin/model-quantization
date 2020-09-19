import math

import numpy as np
import torch.nn as nn

#from compression.models.quantization.dorefa_clip import QConv2d, QLinear
from .quant import custom_conv
from .dorefa_clip import QConv2d, QLinear

__all__ = ["QMobileNetV1"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def qconv3x3(in_planes, out_planes, stride=1, bits_weights=32, bits_activations=32):
    "3x3 convolution with padding"
    return QConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


def conv1x1(in_planes, out_planes, bits_weights=32, bits_activations=32, kernel_size=1, args=None):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


def qconv1x1(in_planes, out_planes, bits_weights=32, bits_activations=32, kernel_size=1, args=None):
    return QConv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


def dwconv3x3(in_planes, out_planes, stride=1, bits_weights=32, bits_activations=32, kernel_size=3, padding=1, groups=1, args=None):
    "3x3 depth wise convolution"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
    )


def qdwconv3x3(in_planes, out_planes, stride=1, bits_weights=32, bits_activations=32, kernel_size=3, padding=1, groups=1, args=None):
    "3x3 depth wise convolution"
    return QConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=in_planes,
        bias=False,
        bits_weights=bits_weights,
        bits_activations=bits_activations,
    )


class QMobileNetV1(nn.Module):
    """
    MobileNetV1 on ImageNet
    """

    def __init__(
        self, num_classes=1000, wide_scale=1.0, bits_weights=32, bits_activations=32, quantize_first_last=False, args=None,
    ):
        super(QMobileNetV1, self).__init__()

        self.args = args
        if self.args is not None and hasattr(self.args, 'fm_bit') and hasattr(self.args, 'fm_enable'):
            if self.args.fm_enable:
                bits_activations = self.args.fm_bit
            if self.args.wt_enable:
                bits_weights = self.args.wt_bit
        # define network structure

        self.layer_width = np.array([32, 64, 128, 256, 512, 1024])
        self.layer_width = np.around(self.layer_width * wide_scale)
        self.layer_width = self.layer_width.astype(int)
        self.segment_layer = [1, 1, 2, 2, 6, 1]  # number of layers in each segment
        self.down_sample = [
            1,
            2,
            3,
            4,
        ]  # the place of down_sample, related to segment_layer

        self.features = self._make_layer(
            bits_weights=bits_weights, bits_activations=bits_activations, quantize_first_last=quantize_first_last
        )
        self.dropout = nn.Dropout(0)
        if quantize_first_last:
            self.classifier = QLinear(self.layer_width[-1].item(), num_classes, bits_weights=8, bits_activations=8)
        else:
            self.classifier = nn.Linear(self.layer_width[-1].item(), 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, bits_weights=32, bits_activations=32, quantize_first_last=False):
        if self.args is not None and hasattr(self.args, 'keyword') and 'lsq' in self.args.keyword:
            qdepth = custom_conv
            qpoint = custom_conv
        elif int(bits_weights) == 32 and int(bits_activations) == 32:
            qdepth = dwconv3x3
            qpoint = conv1x1
        else:
            qdepth = qdwconv3x3
            qpoint = qconv1x1

        if quantize_first_last:
            layer_list = [
                qconv3x3(in_planes=3, out_planes=self.layer_width[0].item(), stride=2, bits_weights=8, bits_activations=32),
                nn.BatchNorm2d(self.layer_width[0].item()),
                nn.ReLU(inplace=True),
            ]
        else:
            layer_list = [
                conv3x3(in_planes=3, out_planes=self.layer_width[0].item(), stride=2),
                nn.BatchNorm2d(self.layer_width[0].item()),
                nn.ReLU(inplace=True),
            ]
        for i, layer_num in enumerate(self.segment_layer):
            in_planes = self.layer_width[i].item()
            for j in range(layer_num):
                if j == layer_num - 1 and i < len(self.layer_width) - 1:
                    out_planes = self.layer_width[i + 1].item()
                else:
                    out_planes = in_planes
                if i in self.down_sample and j == layer_num - 1:
                    stride = 2
                else:
                    stride = 1
                layer_list.append(
                    qdepth(
                    #custom_conv(
                        in_planes,
                        in_planes,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        groups=in_planes,
                        args=self.args,
                        bits_weights=bits_weights,
                        bits_activations=bits_activations,
                    )
                )
                layer_list.append(nn.BatchNorm2d(in_planes))
                layer_list.append(nn.ReLU(inplace=True))
                layer_list.append(
                    qpoint(
                    #custom_conv(
                        in_planes,
                        out_planes,
                        kernel_size=1,
                        args=self.args,
                        bits_weights=bits_weights,
                        bits_activations=bits_activations,
                    )
                )
                layer_list.append(nn.BatchNorm2d(out_planes))
                layer_list.append(nn.ReLU(inplace=True))
                in_planes = out_planes
        layer_list.append(nn.AvgPool2d(7))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        """
        forward propagation
        """
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)

        return out

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.models.mobilenet import _make_divisible

from .base import BasePoint, initialize_weights


# from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig, SqueezeExcitation
# - as of 2021-02-02, not yet available through conda channel (part of version 0.9, only 0.82 available)
# - copied from
#   - https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
#   - https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py
# =>

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result

# <=
# copied part ends


class InvertedPartialResidual(InvertedResidual):
    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        c = min(input.shape[1], result.shape[1])
        result[:, :c, :, :] += input[:, :c, :, :]
        return result


class MobileAP(BasePoint):
    def __init__(self, arch, in_channels=1, batch_norm=True, det_hidden_ch=128, qlt_hidden_ch=128, des_hidden_ch=128,
                 descriptor_dim=128, width_mult=1.0, dropout=0.0, pretrained=False, cache_dir=None):
        super(MobileAP, self).__init__()

        self.conf = {
            'arch': arch,
            'in_channels': in_channels,
            'det_hidden_ch': det_hidden_ch,
            'qlt_hidden_ch': qlt_hidden_ch,
            'des_hidden_ch': des_hidden_ch,
            'descriptor_dim': descriptor_dim,
            'batch_norm': batch_norm,
            'width_mult': width_mult,
            'pretrained': pretrained,
        }

        self.backbone, out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                     width_mult=width_mult, in_channels=in_channels)

        self.des_head = self.create_descriptor_head(out_ch, des_hidden_ch, descriptor_dim, batch_norm, dropout)
        self.det_head = self.create_detector_head(out_ch, det_hidden_ch, batch_norm, dropout)
        self.qlt_head = self.create_quality_head(out_ch, qlt_hidden_ch, batch_norm, dropout)

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization
            init_modules = [self.backbone, self.des_head, self.det_head, self.qlt_head]
            initialize_weights(init_modules)

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1, **kwargs):

        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        if arch[-3:] == 'prn':
            # idea from https://github.com/WongKinYiu/PartialResidualNetworks
            block = partial(InvertedPartialResidual, norm_layer=norm_layer)
        else:
            block = partial(InvertedResidual, norm_layer=norm_layer)

        def add_layer(l, in_ch, kernel, exp_coef, out_ch, use_se, activation, stride, dilation):
            l.append(block(InvertedResidualConfig(in_ch, kernel, round(in_ch*exp_coef), out_ch, use_se, activation,
                                                  stride, dilation, width_mult)))
            return out_ch

        layers = []

        # building first layer
        in_ch = 16
        layers.append(ConvBNActivation(in_channels, in_ch, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # TODO: double check architecture from paper, figure out how to modify

        if arch[:5] == 'large':
            # in_ch, kernel, exp_ch, out_ch, use_se, activation, stride, dilation
            in_ch = add_layer(layers, in_ch, 3, 1, 16, False, "RE", 1, 1)
            in_ch = add_layer(layers, in_ch, 3, 4, 24, False, "RE", 2, 1)   # C1
            in_ch = add_layer(layers, in_ch, 3, 3, 24, False, "RE", 1, 1)
            in_ch = add_layer(layers, in_ch, 5, 3, 40, True, "RE", 2, 1)    # C2
            in_ch = add_layer(layers, in_ch, 5, 3, 40, True, "RE", 1, 1)
            in_ch = add_layer(layers, in_ch, 5, 3, 40, True, "RE", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 6, 80, False, "HS", 2, 1)  # C3
            # in_ch = add_layer(layers, in_ch, 3, 2.5, 80, False, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 2.3, 80, False, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 2.3, 80, False, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 6, 112, True, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 6, 112, True, "HS", 1, 1)
        else:
            in_ch = add_layer(layers, in_ch, 3, 1, 16, True, "RE", 2, 1),  # C1
            in_ch = add_layer(layers, in_ch, 3, 4, 24, False, "RE", 2, 1),  # C2
            in_ch = add_layer(layers, in_ch, 3, 3.67, 24, False, "RE", 1, 1),
            # in_ch = add_layer(layers, in_ch, 5, 4, 40, True, "HS", 2, 1),  # C3
            # in_ch = add_layer(layers, in_ch, 5, 6, 40, True, "HS", 1, 1),
            # in_ch = add_layer(layers, in_ch, 5, 6, 40, True, "HS", 1, 1),
            # in_ch = add_layer(layers, in_ch, 5, 3, 48, True, "HS", 1, 1),
            # in_ch = add_layer(layers, in_ch, 5, 3, 48, True, "HS", 1, 1),

        # TODO: use this for heads?
        # building last several layers
        layers.append(ConvBNActivation(in_ch, 6 * in_ch, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish))

        return nn.Sequential(*layers), 6 * in_ch

    @staticmethod
    def create_detector_head(in_channels, single=False):
        return nn.Conv2d(in_channels, 1 if single else 2, kernel_size=1, padding=0)

    @staticmethod
    def create_quality_head(in_channels, single=False):
        return nn.Conv2d(in_channels, 1 if single else 2, kernel_size=1, padding=0)

    def extract_features(self, x):
        x_features = self.backbone(x)
        x_features, *auxs = x_features if isinstance(x_features, tuple) else (x_features, )
        return [x_features] + list(auxs)

    def forward(self, input):
        # input is a pair of images
        x_features, *auxs = self.extract_features(input)
        x_des = x_features
        x_feat2 = x_features.pow(2)
        x_det = self.det_head(x_feat2)
        x_qlt = self.qlt_head(x_feat2)
        x_output = [self.fix_output(x_des, x_det, x_qlt)]
        exec_aux = self.training and len(auxs) > 0

        if exec_aux:
            for i, aux in enumerate(auxs):
                a_des = aux
                a_feat2 = aux.pow(2)
                a_det = getattr(self, 'aux_t' + str(i + 1))(a_feat2)
                a_qlt = getattr(self, 'aux_q' + str(i + 1))(a_feat2)
                ao = self.fix_output(a_des, a_det, a_qlt)
                x_output.append(ao)

        return x_output if exec_aux else x_output[0]

    @staticmethod
    def activation(ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, :1, :, :]

    def fix_output(self, descriptors, detection, quality):
        des = F.normalize(descriptors, p=2, dim=1)
        det = self.activation(detection)
        qlt = self.activation(quality)
        return des, det, qlt

from functools import partial
from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .base import BasePoint, initialize_weights

try:
    from torchvision.models.mobilenet import _make_divisible
except:
    from torchvision.models.mobilenetv2 import _make_divisible


# from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig, SqueezeExcitation
# - as of 2021-02-02, not yet available through conda channel (part of version 0.9, only 0.82 available)
# - on 2021-02-02 copied from
#   - https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
#   - https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py
# - later small changes, e.g. all nn.ReLU => nn.ReLU6
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

        if False and kernel_size == 5:
            # This part was modified as I wanted to test what the impact would be of splitting 5x5 conv to two 3x3 convs
            conv_layer = nn.Sequential(
                nn.Conv2d(in_planes, in_planes,  3, stride, 1, dilation=dilation, groups=groups, bias=False),
                nn.Conv2d(in_planes, out_planes, 3,      1, 1, dilation=dilation, groups=groups, bias=False),
            )
        else:
            conv_layer = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                                   dilation=dilation, groups=groups, bias=False)

        super(ConvBNActivation, self).__init__(conv_layer, norm_layer(out_planes), activation_layer(inplace=True))
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4, lightweight: bool = True, residual: bool = False):
        super().__init__()
        self.lightweight = lightweight
        self.residual = residual
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = (nn.ReLU6 if lightweight else nn.ReLU)(inplace=True)     # NOTE: change from ReLU to ReLU6 !!!
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=inplace) if self.lightweight else F.sigmoid(scale)
        return scale

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input + (input if self.residual else 0)


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
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU6         # NOTE: change from ReLU to ReLU6 !!!

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

        _, c0, h0, w0 = input.shape
        _, c1, h1, w1 = result.shape
        c = min(c0, c1)

        if h0 == h1 and w0 == w1:
            result[:, :c, :, :] += input[:, :c, :, :]
        else:
            result[:, :c, :, :] += F.interpolate(input[:, :c, :, :], (h1, w1), mode='area')

        return result


# TODO: rename this class to LAFE and this file to lafe.py, will likely cause problems with previously trained models
class MobileAP(BasePoint):
    def __init__(self, arch, des_head, det_head, qlt_head, in_channels=1, partial_residual=False,
                 width_mult=1.0, pretrained=False, cache_dir=None):

        super(MobileAP, self).__init__()

        self.conf = {
            'arch': arch,
            'in_channels': in_channels,
            'partial_residual': partial_residual,
            'width_mult': width_mult,
            'pretrained': pretrained,
            'des_head': des_head,
            'det_head': det_head,
            'qlt_head': qlt_head,
            # 'det_hidden_ch': det_hidden_ch,
            # 'qlt_hidden_ch': qlt_hidden_ch,
            # 'des_hidden_ch': des_hidden_ch,
            # 'descriptor_dim': descriptor_dim,
            # 'dropout': dropout,
            # 'head_exp_coef': head_exp_coef,
            # 'head_use_se': head_use_se,
        }

        # NOTE: Affine/renorm is False by default in TensorFlow, which is used for MobileNet and EfficientNet.
        #       However, in TorchVision models, affine is set to True as in PyTorch its True by default.
        #       Also, PyTorch momentum is 1 - TensorFlow momentum. PyTorch default is 0.1, however,
        #       both mobilenet and efficientnet use 0.01 for momentum.
        self.norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01, affine=False)
        if partial_residual:
            # idea from https://github.com/WongKinYiu/PartialResidualNetworks
            # -- doesn't seem to work well though
            self.block_cls = partial(InvertedPartialResidual, norm_layer=self.norm_layer)
        else:
            self.block_cls = partial(InvertedResidual, norm_layer=self.norm_layer)

        self.backbone, bb_out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                        width_mult=width_mult, in_channels=in_channels)

        self.des_head = self.create_descriptor_head(bb_out_ch, des_head)

        out_ch = des_head['dimensions'] if det_head['after_des'] else bb_out_ch
        self.det_head = self.create_detector_head(out_ch, det_head)

        out_ch = des_head['dimensions'] if qlt_head['after_des'] else bb_out_ch
        self.qlt_head = None if qlt_head['skip'] else self.create_quality_head(out_ch, qlt_head)

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization
            init_modules = [self.backbone, self.des_head, self.det_head, self.qlt_head]
            initialize_weights(init_modules)

    def add_layer(self, l, in_ch, kernel, exp_coef, out_ch, use_se, activation, stride, dilation,
                  force_out_ch=False, force_in_ch=False):
        irc = InvertedResidualConfig(in_ch, kernel, round(in_ch * exp_coef), out_ch, use_se, activation,
                                     stride, dilation, self.conf['width_mult'])
        if force_in_ch:
            irc.input_channels = in_ch
        if force_out_ch:
            irc.out_channels = out_ch
        l.append(self.block_cls(irc))
        return round(irc.out_channels / self.conf['width_mult'])

    def create_backbone(self, arch, cache_dir=None, pretrained=False, in_channels=1, **kwargs):
        arch = arch.lower().split('_')
        a0, a1 = arch[0], arch[1] if len(arch) > 1 else ''

        # building first layer
        in_ch = {'en': 32, 'mn2': 32 if a1 == 'l' else 24, 'mn3': 16}[a0]
        layers = [ConvBNActivation(in_channels, round(in_ch * self.conf['width_mult']), kernel_size=3, stride=2,
                                   norm_layer=self.norm_layer, activation_layer=nn.Hardswish)]

        if a0 == 'mn3' and a1 in ('o', 'l'):
            # mobilenetv3 large

            # in_ch, kernel, exp_ch, out_ch, use_se, activation, stride, dilation
            in_ch = self.add_layer(layers, in_ch, 3, 1, 16, False, "RE", 1, 1)
            in_ch = self.add_layer(layers, in_ch, 3, 4, 24, False, "RE", 2, 1)   # C1
            in_ch = self.add_layer(layers, in_ch, 3, 3, 24, False, "RE", 1, 1)
            in_ch = self.add_layer(layers, in_ch, 5, 3, 40, True, "RE", 2, 1)    # C2
            in_ch = self.add_layer(layers, in_ch, 5, 3, 40, True, "RE", 1, 1)
            in_ch = self.add_layer(layers, in_ch, 5, 3, dict(o=40, l=72)[a1], True, "RE", 1, 1)  # hf-net mod: 40=>64, or 72?
            # in_ch = add_layer(layers, in_ch, 3, 6, 80, False, "HS", 2, 1)  # C3
            # in_ch = add_layer(layers, in_ch, 3, 2.5, 80, False, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 2.3, 80, False, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 2.3, 80, False, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 6, 112, True, "HS", 1, 1)
            # in_ch = add_layer(layers, in_ch, 3, 6, 112, True, "HS", 1, 1)
        elif a0 == 'en' and a1 in ('0', 'l'):
            # EfficientNet-B0 from https://arxiv.org/pdf/1905.11946v5.pdf

            # in_ch, kernel, exp_ch, out_ch, use_se, activation, stride, dilation
            in_ch = self.add_layer(layers, in_ch, 3, 1, 16, True, "HS", 1, 1)
            in_ch = self.add_layer(layers, in_ch, 3, 6, 24, True, "HS", 2, 1)  # C1
            in_ch = self.add_layer(layers, in_ch, 3, 6, 24, True, "HS", 1, 1)
            in_ch = self.add_layer(layers, in_ch, 5, 6, 40, True, "HS", 2, 1)  # C2
            in_ch = self.add_layer(layers, in_ch, 5, 6, {'0': 40, 'l': 72}[a1], True, "HS", 1, 1)  # hf-net mod: 40=>64, or 72?

        elif a0 == 'mn2' and a1 in ('o', 'm', 'l'):
            # mobilenet v2
            # however, use arch from the code of hf-net instead, article version is yet a bit different:
            # https://arxiv.org/abs/1812.03506

            # in_ch, kernel, exp_ch, out_ch, use_se, activation, stride, dilation
            in_ch = self.add_layer(layers, in_ch, 3, 1, 16, False, "RE", 1, 1)
            in_ch = self.add_layer(layers, in_ch, 3, 6, 24, False, "RE", 2, 1)   # C1
            in_ch = self.add_layer(layers, in_ch, 3, 6, 24, False, "RE", 1, 1)
            in_ch = self.add_layer(layers, in_ch, 3, 6, 32, False, "RE", 2, 1)   # C2
            in_ch = self.add_layer(layers, in_ch, 3, 6, dict(o=32, m=32, l=64)[a1], False, "RE", 1, 1)   # hf-net mod: 32=>64
            in_ch = self.add_layer(layers, in_ch, 3, 6, dict(o=32, m=64, l=128)[a1], False, "RE", 1, 1)  # hf-net mod: 32=>128

        else:
            assert a0 == 'mn3' and a1 == 's', 'invalid arch %s' % (arch,)
            # mobilenetv3 small
            in_ch = self.add_layer(layers, in_ch, 3, 1, 16, True, "RE", 2, 1)   # C1
            in_ch = self.add_layer(layers, in_ch, 3, 4, 24, False, "RE", 2, 1)  # C2
            # in_ch = self.add_layer(layers, in_ch, 3, 3.67, 24, False, "RE", 1, 1),
            in_ch = self.add_layer(layers, in_ch, 5, 4, 40, True, "RE", 1, 1)       # hf-net mod: instead of above line
            # in_ch = add_layer(layers, in_ch, 5, 4, 40, True, "HS", 2, 1),  # C3
            # in_ch = add_layer(layers, in_ch, 5, 6, 40, True, "HS", 1, 1),
            # in_ch = add_layer(layers, in_ch, 5, 6, 40, True, "HS", 1, 1),
            # in_ch = add_layer(layers, in_ch, 5, 3, 48, True, "HS", 1, 1),
            # in_ch = add_layer(layers, in_ch, 5, 3, 48, True, "HS", 1, 1),

        return nn.Sequential(*layers), round(in_ch * self.conf['width_mult'])

    def _create_head(self, in_ch, out_ch, conf):
        seq = []
        if conf['hidden_ch'] > 0:
            if conf['exp_coef'] > 0:
                self.add_layer(seq, in_ch, 3, conf['exp_coef'], conf['hidden_ch'], conf['use_se'], "HS", 1, 1,
                               force_out_ch=True, force_in_ch=True)
                in_ch = conf['hidden_ch']
            else:
                seq.append(nn.Conv2d(in_ch, conf['hidden_ch'], kernel_size=3, padding=1))
                seq.append(nn.BatchNorm2d(conf['hidden_ch']))
                # TODO: fix the current hack where squeeze-exitation param is used to determine activation function
                seq.append(nn.Hardswish() if conf['use_se'] else nn.ReLU6())
                in_ch = conf['hidden_ch']

        if conf['dropout'] > 0:
            seq.append(nn.Dropout(conf['dropout']))

        seq.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
        return nn.Sequential(*seq)

    def create_descriptor_head(self, in_ch, conf):
        return self._create_head(in_ch, conf['dimensions'], conf)

    def create_detector_head(self, in_ch, conf):
        if conf['act_fn_type'] == 'softmax':
            return self._create_head(in_ch, 65, conf)
        else:
            return self._create_head(in_ch, 64, conf)

    def create_quality_head(self, in_channels, conf):
        out_ch = 1 if conf.get('single', True) else 2
        return self._create_head(in_channels, out_ch, conf)

    def forward(self, input):
        # input is a pair of images
        features = self.backbone(input)

        des = self.des_head(features)
        des2 = None

        if self.conf['det_head']['after_des']:
            des2 = des.pow(2) if des2 is None else des2
            det = self.det_head(des2)
        else:
            det = self.det_head(features)

        if self.qlt_head is None:
            qlt = None
        elif self.conf['qlt_head']['after_des']:
            des2 = des.pow(2) if des2 is None else des2
            qlt = self.qlt_head(des2)
        else:
            qlt = self.qlt_head(features)

        output = self.fix_output(des, det, qlt)
        return output

    def fix_output(self, descriptors, detection, quality):
        des = F.normalize(descriptors, p=2, dim=1)

        if self.conf['det_head']['act_fn_type'] == 'softmax':
            det = F.pixel_shuffle(F.softmax(detection, dim=1)[:, 1:, :, :], 8)
        else:
            det = F.pixel_shuffle(detection, 8)
            det = self.activation(det, fn_type=self.conf['det_head']['act_fn_type'])

        if quality is None:
            qlt = torch.ones_like(det)
        else:
            qlt = self.activation(quality, fn_type=self.conf['qlt_head']['act_fn_type'])

        return des, det, qlt

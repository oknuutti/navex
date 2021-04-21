import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan

from .base import BasePoint, initialize_weights

LINEAR_QLT = False


class R2D2(BasePoint):
    def __init__(self, arch, des_head, det_head, qlt_head, in_channels=1,
                 width_mult=1.0, pretrained=False, cache_dir=None):
        super(R2D2, self).__init__()

        self.conf = {
            'arch': arch,
            'in_channels': in_channels,
            'descriptor_dim': des_head['dimensions'],
            'width_mult': width_mult,
            'pretrained': pretrained,
        }

        assert width_mult * 128 == des_head['dimensions'], 'descriptor dimensions dont correspond with backbone width'
        self.backbone, out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                     width_mult=width_mult, in_channels=in_channels)
        assert out_ch == des_head['dimensions'], 'channel depths dont match'

        # det_head single=True in r2d2 github code, in article was single=False though
        self.det_head = self.create_detector_head(des_head['dimensions'], single=True)
        self.qlt_head = self.create_quality_head(des_head['dimensions'], single=True)

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization
            initialize_weights([self.backbone, self.det_head])

            # kaiming initialization but x100 lower gain than normal
            if LINEAR_QLT:
                fan = _calculate_correct_fan(self.qlt_head.weight, 'fan_in')
                std = 0.01 / math.sqrt(fan)  # normal gain for linear nonlinearity would be `1`
                base = -math.log(2.0)
                with torch.no_grad():
                    nn.init.normal_(self.qlt_head.weight, 0, std)
                    nn.init.constant_(self.qlt_head.bias, base)
            else:
                initialize_weights([self.qlt_head])

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1, **kwargs):
        def add_layer(l, in_ch, out_ch, k=3, p=1, d=1, bn=True, relu=True):
            out_ch = int(out_ch)
            l.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, dilation=d))
            if bn:
                l.append(nn.BatchNorm2d(out_ch, affine=False))
            if relu:
                l.append(nn.ReLU(inplace=True))
            return out_ch

        layers = []
        wm, in_ch = int(4*width_mult), in_channels
        in_ch = add_layer(layers, in_ch, 8 * wm)
        in_ch = add_layer(layers, in_ch, 8 * wm)
        in_ch = add_layer(layers, in_ch, 16 * wm)
        in_ch = add_layer(layers, in_ch, 16 * wm, p=2, d=2)
        in_ch = add_layer(layers, in_ch, 32 * wm, p=2, d=2)
        in_ch = add_layer(layers, in_ch, 32 * wm, p=4, d=4)
        in_ch = add_layer(layers, in_ch, 32 * wm, p=2, d=4, k=2, relu=False)
        in_ch = add_layer(layers, in_ch, 32 * wm, p=4, d=8, k=2, relu=False)
        in_ch = add_layer(layers, in_ch, 32 * wm, p=8, d=16, k=2, relu=False, bn=False)

        return nn.Sequential(*layers), in_ch

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
            return F.softmax(ux, dim=1)[:, 1:2, :, :]   # was long time ":1" instead of "1:2" in own implementation

    def fix_output(self, descriptors, detection, quality):
        des = F.normalize(descriptors, p=2, dim=1)
        det = self.activation(detection)
        qlt = quality if LINEAR_QLT else self.activation(quality)
        return des, det, qlt

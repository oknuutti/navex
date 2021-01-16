
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePoint, initialize_weights


class R2D2(BasePoint):
    def __init__(self, arch, in_channels=1, batch_norm=True, descriptor_dim=128,
                 width_mult=1.0, pretrained=False, excl_bn_affine=True, cache_dir=None):
        super(R2D2, self).__init__()

        self.conf = {
            'arch': arch,
            'in_channels': in_channels,
            'batch_norm': batch_norm,
            'descriptor_dim': descriptor_dim,
            'width_mult': width_mult,
            'pretrained': pretrained,
            'excl_bn_affine': excl_bn_affine,
        }

        self.backbone, out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                     width_mult=width_mult, batch_norm=batch_norm, type='r2d2',
                                                     in_channels=in_channels, depth=3)

        # det_head single=True in r2d2 github code, in article was single=False though
        self.det_head = self.create_detector_head(descriptor_dim, single=True)
        self.qlt_head = self.create_quality_head(descriptor_dim)

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization (backbone already initialized in its __init__ method)
            init_modules = [self.det_head, self.qlt_head]
            initialize_weights(init_modules)

        if excl_bn_affine:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.affine = False
                    nn.init.constant_(m.bias.data, 0)
                    nn.init.constant_(m.weight.data, 1)

    @staticmethod
    def create_detector_head(in_channels, single=False):
        return nn.Conv2d(in_channels, 2, kernel_size=1, padding=0)

    @staticmethod
    def create_quality_head(in_channels, single=False):
        return nn.Conv2d(in_channels, 2, kernel_size=1, padding=0)

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
            return F.softmax(ux, dim=1)[:, 1:2]

    def fix_output(self, descriptors, detection, quality):
        des = F.normalize(descriptors, p=2, dim=1)
        det = self.activation(detection)
        qlt = self.activation(quality)
        return des, det, qlt

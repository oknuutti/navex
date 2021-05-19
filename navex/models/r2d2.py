
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePoint, initialize_weights


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
            'det_head': det_head,
            'qlt_head': qlt_head,
        }
        separate_des_head = not det_head['after_des'] or (not qlt_head['after_des'] and not qlt_head['skip'])

        assert width_mult * 128 == des_head['dimensions'], 'descriptor dimensions dont correspond with backbone width'
        self.backbone, bb_out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                        width_mult=width_mult, in_channels=in_channels,
                                                        separate_des_head=separate_des_head)

        if separate_des_head:
            self.des_head = self.create_descriptor_head(bb_out_ch, des_head['dimensions'])
        else:
            assert bb_out_ch == des_head['dimensions'], 'channel depths dont match'
            self.des_head = None

        # det_head single=True in r2d2 github code, in article was single=False though
        out_ch = des_head['dimensions'] if det_head['after_des'] else bb_out_ch
        self.det_head = self.create_detector_head(out_ch, single=True)

        out_ch = des_head['dimensions'] if qlt_head['after_des'] else bb_out_ch
        self.qlt_head = None if qlt_head['skip'] else \
                        self.create_quality_head(out_ch, single=qlt_head.get('single', True))

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization
            initialize_weights([self.backbone, self.det_head, self.qlt_head])

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1,
                        separate_des_head=False, **kwargs):
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
        in_ch = add_layer(layers, in_ch, 32 * wm, p=8, d=16, k=2, relu=False, bn=separate_des_head)

        return nn.Sequential(*layers), in_ch

    @staticmethod
    def create_descriptor_head(in_ch, out_ch):
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    @staticmethod
    def create_detector_head(in_ch, single=False):
        return nn.Conv2d(in_ch, 1 if single else 2, kernel_size=1, padding=0)

    @staticmethod
    def create_quality_head(in_ch, single=False):
        return nn.Conv2d(in_ch, 1 if single else 2, kernel_size=1, padding=0)

    def forward(self, input):
        # input is a pair of images
        features = self.backbone(input)

        des = features if getattr(self, 'des_head', None) is None else self.des_head(features)
        des2 = None

        if 'det_head' not in self.conf or self.conf['det_head']['after_des']:
            des2 = des.pow(2) if des2 is None else des2
            det = self.det_head(des2)
        else:
            det = self.det_head(features)

        if self.qlt_head is None:
            qlt = None
        elif 'qlt_head' not in self.conf or self.conf['qlt_head']['after_des']:
            des2 = des.pow(2) if des2 is None else des2
            qlt = self.qlt_head(des2)
        else:
            qlt = self.qlt_head(features)

        output = self.fix_output(des, det, qlt)
        return output

    @staticmethod
    def activation(ux, T=1):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)
        elif ux.shape[1] == 2:
            # T is for temperature scaling, referred e.g. at https://arxiv.org/pdf/1706.04599.pdf
            return F.softmax(ux/T, dim=1)[:, 1:2, :, :]   # was long time ":1" instead of "1:2" in own implementation

    def fix_output(self, descriptors, detection, quality):
        des = F.normalize(descriptors, p=2, dim=1)
        det = self.activation(detection)

        # could use eval_T=100, however, had worse performance, useful possibly for analyzing quality output
        eval_T = 1
        qlt = det if quality is None else self.activation(quality, T=1 if self.training else eval_T)

        return des, det, qlt

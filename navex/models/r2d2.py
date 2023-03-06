
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePoint, initialize_weights


class R2D2(BasePoint):
    def __init__(self, arch, des_head, det_head, qlt_head, in_channels=1, train_with_raw_act_fn=False,
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
            'train_with_raw_act_fn': train_with_raw_act_fn,
        }
        self.separate_des_head = not det_head['after_des'] or (not qlt_head['after_des'] and not qlt_head['skip'])

        assert width_mult * 128 == des_head['dimensions'], 'descriptor dimensions dont correspond with backbone width'
        self.backbone, bb_out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                        width_mult=width_mult, in_channels=in_channels,
                                                        separate_des_head=self.separate_des_head)

        if self.separate_des_head:
            self.des_head = self.create_descriptor_head(bb_out_ch, des_head)
        else:
            self.des_head = None

        # det_head single=True in r2d2 github code, in article was single=False though
        out_ch = des_head['dimensions'] if det_head['after_des'] else bb_out_ch
        self.det_head = self.create_detector_head(out_ch, det_head)

        out_ch = des_head['dimensions'] if qlt_head['after_des'] else bb_out_ch
        self.qlt_head = None if qlt_head['skip'] else self.create_quality_head(out_ch, qlt_head)

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization
            initialize_weights([m for m in (self.backbone, self.det_head, self.qlt_head, self.des_head) if m is not None])

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1,
                        separate_des_head=False, **kwargs):
        def add_layer(l, in_ch, out_ch, k=3, d=1, bn=True, relu=True):
            out_ch = int(out_ch)
            p = ((k-1)*d)//2
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
        in_ch = add_layer(layers, in_ch, 16 * wm, d=2)
        in_ch = add_layer(layers, in_ch, 32 * wm, d=2)
        in_ch = add_layer(layers, in_ch, 32 * wm, d=4)
        in_ch = add_layer(layers, in_ch, 32 * wm, d=4, k=2, relu=False)
        in_ch = add_layer(layers, in_ch, 32 * wm, d=8, k=2, relu=False)
        in_ch = add_layer(layers, in_ch, 32 * wm, d=16, k=2, relu=separate_des_head, bn=separate_des_head)

        return nn.Sequential(*layers), in_ch

    @staticmethod
    def _create_head(in_ch, out_ch, conf, hidden_k=1, hidden_g=1):
        seq = []
        if conf['hidden_ch'] > 0:
            if conf['exp_coef'] > 0:
                raise NotImplemented()
            else:
                seq.append(nn.Conv2d(in_ch, conf['hidden_ch'], groups=hidden_g,
                                     kernel_size=hidden_k, padding=hidden_k//2))
                seq.append(nn.BatchNorm2d(conf['hidden_ch']))
                seq.append(nn.ReLU())
                in_ch = conf['hidden_ch']

        if conf['dropout'] > 0:
            seq.append(nn.Dropout(conf['dropout']))

        seq.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
        return nn.Sequential(*seq)

    @classmethod
    def create_descriptor_head(cls, in_ch, conf):
        return cls._create_head(in_ch, conf['dimensions'], conf)

    @classmethod
    def create_detector_head(cls, in_ch, conf):
        return cls._create_head(in_ch, 1, conf, hidden_k=3, hidden_g=conf['hidden_ch'] or 1)

    @classmethod
    def create_quality_head(cls, in_ch, conf):
        out_ch = 1 if conf.get('single', True) else 2
        return cls._create_head(in_ch, out_ch, conf)

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

    def fix_output(self, descriptors, detection, quality):
        des = F.normalize(descriptors, p=2, dim=1)

        no_act_fn = self.training and self.conf['train_with_raw_act_fn']
        det = self.activation(detection,
                              fn_type='none' if no_act_fn else self.conf['det_head'].get('act_fn_type', 'r2d2'))
        # det = F.avg_pool2d(det, 3, stride=1, padding=1)

        # could use eval_T=100, however, had worse performance, useful possibly for analyzing quality output
        eval_T = 1
        if quality is None:
            qlt = det
        else:
            qlt = self.activation(quality, T=1 if self.training else eval_T,
                                  fn_type='none' if no_act_fn else self.conf['qlt_head'].get('act_fn_type', 'r2d2'))

        return des, det, qlt


import torch.nn as nn
import torch.nn.functional as F

from .base import BasePoint, initialize_weights


class AstroPoint(BasePoint):
    def __init__(self, arch, des_head, det_head, qlt_head, in_channels=1,
                 width_mult=1.0, pretrained=False, cache_dir=None):
        super(AstroPoint, self).__init__()

        self.conf = {
            'arch': arch,
            'in_channels': in_channels,
            'width_mult': width_mult,
            'pretrained': pretrained,
            'des_head': des_head,
            'det_head': det_head,
            'qlt_head': qlt_head,
        }

        self.backbone, out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                     width_mult=width_mult, subtype='sp',
                                                     in_channels=in_channels, depth=3)

        self.des_head = self.create_descriptor_head(out_ch, des_head)
        self.det_head = self.create_detector_head(out_ch, det_head)
        self.qlt_head = self.create_quality_head(out_ch, qlt_head)

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization (backbone already initialized in its __init__ method)
            init_modules = [self.des_head, self.det_head, self.qlt_head]
            initialize_weights(init_modules)

        # don't use affine transform for batch norm
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
                m.affine = False
                nn.init.constant_(m.bias.data, 0)
                nn.init.constant_(m.weight.data, 1)

    @staticmethod
    def _create_head(in_ch, out_ch, conf):
        seq = []
        if conf['hidden_ch'] > 0:
            seq.append(nn.Conv2d(in_ch, conf['hidden_ch'], kernel_size=3, padding=1))
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
        return cls._create_head(in_ch, 65, conf)

    @classmethod
    def create_quality_head(cls, in_ch, conf):
        return cls._create_head(in_ch, 2, conf)

    def params_to_optimize(self, split=False):
        np = list(self.named_parameters(recurse=False))
        names, params = zip(*np) if len(np) > 0 else ([], [])
        bnclss = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d

        for mn, m in self.named_modules():
            # maybe exclude all params from BatchNorm layers
            if isinstance(m, bnclss) and m.affine or not isinstance(m, bnclss):
                np = list(m.named_parameters(recurse=False))
                n, p = zip(*np) if len(np) > 0 else ([], [])
                names.extend([mn+'.'+k for k in n])
                params.extend(p)

        new_biases, new_weights, biases, weights, others = [], [], [], [], []
        for name, param in zip(names, params):
            is_new = ('des_head' in name or 'det_head' in name or 'qlt_head' in name or 'aux' in name)
            if is_new and 'bias' in name:
                new_biases.append(param)
            elif is_new and 'weight' in name:
                new_weights.append(param)
            elif 'bias' in name:
                biases.append(param)
            elif 'weight' in name:
                weights.append(param)
            else:
                others.append(param)

        return (new_biases, new_weights, biases, weights, others) if split \
                else (new_biases + new_weights + biases + weights + others)

    def extract_features(self, x):
        x_features = self.backbone(x)
        x_features, *auxs = x_features if isinstance(x_features, tuple) else (x_features, )
        return [x_features] + list(auxs)

    def forward(self, input):
        # input is a pair of images
        x_features, *auxs = self.extract_features(input)
        x_des = self.des_head(x_features)
        x_det = self.det_head(x_features)
        x_qlt = self.qlt_head(x_features)
        x_output = [self.fix_output(x_des, x_det, x_qlt)]
        exec_aux = self.training and len(auxs) > 0

        if exec_aux:
            for i, aux in enumerate(auxs):
                a_des = getattr(self, 'aux_s' + str(i + 1))(aux)
                a_det = getattr(self, 'aux_t' + str(i + 1))(aux)
                a_qlt = getattr(self, 'aux_q' + str(i + 1))(aux)
                ao = self.fix_output(a_des, a_det, a_qlt)
                x_output.append(ao)

        return x_output if exec_aux else x_output[0]

    def fix_output(self, descriptors, detection, quality):
        des = F.normalize(descriptors, p=2, dim=1)
        detection = F.pixel_shuffle(F.softmax(detection, dim=1)[:, 1:, :, :], 8)

        # like in R2D2
        def activation(x):
            if x.shape[1] == 1:
                x = F.softplus(x)
                return x / (1 + x)
            else:
                return F.softmax(x, dim=1)[:, :1, :, :]

        det = activation(detection)
        qlt = activation(quality)
        return des, det, qlt


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

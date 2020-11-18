
import torch.nn as nn
import torch.nn.functional as F

from .base import BasePoint, initialize_weights


class AstroPoint(BasePoint):
    def __init__(self, arch, head_conv_ch=256, direct_detection=False, batch_norm=True, descriptor_dim=256, width_mult=1.0,
                 dropout=0.0, pretrained=False, excl_bn_affine=False, cache_dir=None):
        super(AstroPoint, self).__init__()

        self.conf = {
            'arch': arch,
            'head_conv_ch': head_conv_ch,
            'direct_detection': direct_detection,
            'batch_norm': batch_norm,
            'descriptor_dim': descriptor_dim,
            'width_mult': width_mult,
            'dropout': dropout,
            'pretrained': pretrained,
            'excl_bn_affine': excl_bn_affine,
        }

        self.backbone, out_ch = self.create_backbone(arch=arch, cache_dir=cache_dir, pretrained=pretrained,
                                                     width_mult=width_mult, batch_norm=batch_norm)

        self.des_head = self.create_descriptor_head(out_ch, descriptor_dim, head_conv_ch>0, batch_norm, dropout)
        self.det_head = self.create_detector_head(out_ch, head_conv_ch, direct_detection, batch_norm, dropout)
        self.qlt_head = self.create_quality_head(out_ch, head_conv_ch, batch_norm, dropout)

        if pretrained:
            raise NotImplemented()
        else:
            # Initialization (backbone already initialized in its __init__ method)
            init_modules = [self.des_head, self.det_head, self.qlt_head]
            initialize_weights(init_modules)

        if excl_bn_affine:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    nn.init.constant_(m.bias.data, 0)
                    nn.init.constant_(m.weight.data, 1)

    @staticmethod
    def create_descriptor_head(in_channels, out_channels, pre_conv=True, batch_norm=False, dropout=0.0):
        seq = []
        if pre_conv:
            seq.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                seq.append(nn.BatchNorm2d(out_channels))
            seq.append(nn.ReLU())
            in_channels = out_channels

        if dropout > 0:
            seq.append(nn.Dropout(dropout))

        seq.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        return nn.Sequential(*seq)

    @staticmethod
    def create_detector_head(in_channels, pre_conv_channels, direct=False, batch_norm=False, dropout=0.0):
        seq = []
        if pre_conv_channels:
            seq.append(nn.Conv2d(in_channels, pre_conv_channels, kernel_size=3, padding=1))
            if batch_norm:
                seq.append(nn.BatchNorm2d(pre_conv_channels))
            seq.append(nn.ReLU())
            in_channels = pre_conv_channels

        if dropout > 0:
            seq.append(nn.Dropout(dropout))

        seq.append(nn.Conv2d(in_channels, 1 if direct else 65, kernel_size=1, padding=0))
        return nn.Sequential(*seq)

    @staticmethod
    def create_quality_head(in_channels, pre_conv_channels, batch_norm=False, dropout=0.0):
        seq = []
        if pre_conv_channels:
            seq.append(nn.Conv2d(in_channels, pre_conv_channels, kernel_size=3, padding=1))
            if batch_norm:
                seq.append(nn.BatchNorm2d(pre_conv_channels))
            seq.append(nn.ReLU())
            in_channels = pre_conv_channels

        if dropout > 0:
            seq.append(nn.Dropout(dropout))

        seq.append(nn.Conv2d(in_channels, 1, kernel_size=1, padding=0))
        return nn.Sequential(*seq)

    def params_to_optimize(self, split=False, excl_batch_norm=True):
        np = list(self.named_parameters(recurse=False))
        names, params = zip(*np) if len(np) > 0 else ([], [])
        for mn, m in self.named_modules():
            # maybe exclude all params from BatchNorm layers
            if not excl_batch_norm or not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
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
        # TODO: check how it actually works with batches
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
        if not self.conf['direct_detection']:
            detection = F.pixel_shuffle(F.softmax(detection, dim=1)[:, 1:, :, :], 8)

        if 1:
            # like in R2D2
            def activation(x):
                x = F.softplus(x)
                return x / (1 + x)
        else:
            # why not?
            activation = F.sigmoid

        det = activation(detection)
        qlt = activation(quality)
        return des, det, qlt


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

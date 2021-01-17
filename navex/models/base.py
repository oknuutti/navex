import os

from torch import nn
import torchvision.models as models

from . import MODELS as own_models


class BasePoint(nn.Module):
    def __init__(self):
        super(BasePoint, self).__init__()
        self.aux_qty = None

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1, **kwargs):
        if cache_dir is not None:
            os.environ['TORCH_HOME'] = cache_dir

        if arch in own_models:
            backbone = own_models[arch]()(pretrained=pretrained, width_mult=width_mult, **kwargs)
            return backbone, backbone.out_channels

        assert arch in models.__dict__, 'invalid model name: %s' % arch

        identity_layers = None   # everything that comes after the output layer, set to Identity-class
        aux_layers = tuple()
        if arch.startswith('alexnet'):
            backbone = getattr(models, arch)(pretrained=pretrained)  # 61M params
            input_layer = 'features.0'
            raise NotImplemented()
            identity_layers = ('classifier',)

        elif arch.startswith('vgg'):
            backbone = getattr(models, arch)(pretrained=pretrained)  # 11&13: 133M, 16: 138M, 19: 144M
            input_layer = 'features.0'
            raise NotImplemented()
            identity_layers = ('classifier',)

        elif arch.startswith('resnet'):
            backbone = getattr(models, arch)(pretrained=pretrained)  # 18: 12M, 34: 22M, 50: 26M, 101: 45M, 152: 60M
            input_layer = 'conv1'
            raise NotImplemented()
            identity_layers = ('fc',)

        elif arch.startswith('squeezenet'):
            backbone = getattr(models, arch)(pretrained=pretrained)  # 1_0 & 1_1: 1.2M
            input_layer = 'features.0'
            raise NotImplemented()
            identity_layers = ('classifier',)

        elif arch.startswith('mobilenet'):
            backbone = getattr(models, arch)(pretrained=pretrained, width_mult=width_mult)  # v2: 3.5M
            input_layer = 'features.0.0'
            raise NotImplemented()
            identity_layers = ('classifier',)

        elif arch.startswith('densenet'):
            backbone = getattr(models, arch)(pretrained=pretrained)  # 121: 8M, 169: 14M, 201: 20M, 161: 29M
            input_layer = 'features.0'  # or features.conv0
            raise NotImplemented()
            identity_layers = ('classifier',)

        elif arch.startswith('inception'):
            backbone = getattr(models, arch)(pretrained=pretrained, aux_logits=True,
                                                           transform_input=False)  # v3: 27M params
            input_layer = 'Conv2d_1a_3x3.conv'
            raise NotImplemented()
            identity_layers = ('fc',)
            aux_layers = ('AuxLogits.fc',)

        elif arch.startswith('googlenet'):
            backbone = getattr(models, arch)(pretrained=pretrained, aux_logits=True,
                                                           transform_input=False)  # 13M params
            input_layer = 'conv1.conv'
            raise NotImplemented()
            identity_layers = ('fc',)
            aux_layers = ('aux1.fc2', 'aux2.fc2')

        else:
            assert False, 'model %s not supported' % arch

        # replace 3-channel input layer with 1-channel gray-scale layer
        parent, name, l = self.dig_layer(input_layer)
        new_input_layer = nn.Conv2d(in_channels=1, out_channels=l.out_channels, kernel_size=l.kernel_size,
                                    stride=l.stride, padding=l.padding, dilation=l.dilation, groups=l.groups,
                                    bias=l.bias)
        setattr(parent, name, new_input_layer)

        # replace all extra layers with identity layers
        for i, identity_layer in enumerate(identity_layers):
            parent, name, layer = self.dig_layer(backbone, identity_layer)
            setattr(parent, name, Identity())
            if i == 0:
                if isinstance(layer, nn.Sequential):
                    for layer in layer:
                        if getattr(layer, 'in_features', False):
                            break
                out_channels = layer.in_channels

        # replace all aux outputs with aux heads
        self.aux_qty = 0
        for i, aux_layer in enumerate(aux_layers):
            parent, name, layer = self.dig_layer(aux_layer)
            setattr(parent, name, Identity())
            setattr(self, 'aux_s'+str(i), self.create_descriptor_head(layer.in_channels))
            setattr(self, 'aux_t'+str(i), self.create_detector_head(layer.in_channels))
            setattr(self, 'aux_q'+str(i), self.create_quality_head(layer.in_channels))
            self.aux_qty = i+1

        # Initialization (backbone already initialized in its __init__ method)
        init_modules = [new_input_layer] \
                       + [getattr(self, 'aux_s' + str(i)) for i in range(1, self.aux_qty + 1)] \
                       + [getattr(self, 'aux_t' + str(i)) for i in range(1, self.aux_qty + 1)] \
                       + [getattr(self, 'aux_q' + str(i)) for i in range(1, self.aux_qty + 1)]
        initialize_weights(init_modules)

        return backbone, out_channels

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

    @staticmethod
    def dig_layer(model, path):
        np = model
        for name in path.split('.'):
            p = np
            np = getattr(p, name)
        return p, name, np


def initialize_weights(modules, std=0.01):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, std)
            nn.init.constant_(m.bias, 0)

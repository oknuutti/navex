
from torch import nn

from .base import initialize_weights


class VGG(nn.Module):

    def __init__(self, pretrained, batch_norm=True, subtype='sp', width_mult=1, in_channels=1, depth=3, **kwargs):
        super(VGG, self).__init__()

        self.model, self.out_channels = make_layers(cfgs[subtype], batch_norm=batch_norm, width_mult=width_mult,
                                                    in_channels=in_channels, depth=depth)

        if pretrained:
            raise NotImplemented()
        else:
            initialize_weights(self.modules())

    def forward(self, x):
        return self.model(x)


def make_layers(cfg, batch_norm=False, width_mult=1, in_channels=1, depth=3):
    layers = []
    k = 0
    out_channels = in_channels
    for c in cfg:
        if c == 'M':
            if k >= depth:
                break
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            k += 1
        else:
            specs = [None, 1, 3]
            for i, v in enumerate(c if isinstance(c, (tuple, list)) else (c,)):
                specs[i] = v
            v, d, s = specs

            out_channels = int(v*width_mult)
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=s, padding=1, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers), out_channels


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'sp': [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128],
}

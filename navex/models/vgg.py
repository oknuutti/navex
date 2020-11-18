
from torch import nn

from .base import initialize_weights


class VGG(nn.Module):

    def __init__(self, pretrained, in_channels=1, batch_norm=True, subtype='sp', width_mult=1, depth=3, **kwargs):
        super(VGG, self).__init__()

        self.model, self.out_channels = \
                make_layers(cfgs[subtype], in_channels=in_channels, batch_norm=batch_norm,
                            width_mult=width_mult, depth=depth)

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
    for v in cfg:
        if v == 'M':
            if k >= depth:
                break
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            k += 1
        else:
            out_channels = int(v*width_mult)
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
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

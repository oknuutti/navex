import torch
from torch import nn

from .r2d2 import R2D2

# test:
#  - affine vs not
#  - 8x8 vs 3x 2x2
#  - hynet vs r2d2 arch


class HyNet(R2D2):
    def __init__(self, arch, des_head, det_head, qlt_head, **kwargs):
        super(HyNet, self).__init__(arch, des_head, det_head, qlt_head, **kwargs)

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1,
                        separate_des_head=False, bn_affine=False, **kwargs):
        def add_layer(l, in_ch, out_ch, k=3, p=1, d=1, frn=True, bn=False):
            out_ch = int(out_ch)
            if frn:
                l.append(FRN(in_ch, affine=bn_affine))
            l.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, dilation=d))
            if bn:
                l.append(nn.BatchNorm2d(out_ch, affine=bn_affine))
            return out_ch

        layers = []
        wm, in_ch = int(4*width_mult), in_channels
        if 0:
            # as in hynet
            in_ch = add_layer(layers, in_ch, 8 * wm)                # 3x3, 32
            in_ch = add_layer(layers, in_ch, 8 * wm)                # 3x3, 32
            in_ch = add_layer(layers, in_ch, 16 * wm, p=2, d=2)     # 3x3, 64, /2
            in_ch = add_layer(layers, in_ch, 16 * wm, p=2, d=2)     # 3x3, 64
            in_ch = add_layer(layers, in_ch, 32 * wm, p=4, d=4)     # 3x3, 128, /2
            in_ch = add_layer(layers, in_ch, 32 * wm, p=4, d=4)     # 3x3, 128
            in_ch = add_layer(layers, in_ch, 32 * wm, p=2, d=4, k=2)             # 2x2, 128 (8x8 1/3)
            in_ch = add_layer(layers, in_ch, 32 * wm, p=4, d=8, k=2, frn=False)  # 2x2, 128 (8x8 2/3)
            in_ch = add_layer(layers, in_ch, 32 * wm, p=8, d=16, k=2, frn=False, bn=separate_des_head)  # 2x2, 128 (8x8 3/3)
        else:
            # as in r2d2
            in_ch = add_layer(layers, in_ch, 8 * wm)
            in_ch = add_layer(layers, in_ch, 8 * wm)
            in_ch = add_layer(layers, in_ch, 16 * wm)
            in_ch = add_layer(layers, in_ch, 16 * wm, p=2, d=2)
            in_ch = add_layer(layers, in_ch, 32 * wm, p=2, d=2)
            in_ch = add_layer(layers, in_ch, 32 * wm, p=4, d=4)
            in_ch = add_layer(layers, in_ch, 32 * wm, p=2, d=4, k=2)
            in_ch = add_layer(layers, in_ch, 32 * wm, p=4, d=8, k=2, frn=False)
            in_ch = add_layer(layers, in_ch, 32 * wm, p=8, d=16, k=2, frn=False, bn=separate_des_head)

        return nn.Sequential(*layers), in_ch


class FRN(nn.Module):
    def __init__(self, channels, init_tau=-1, eps=1e-6, affine=True):
        super(FRN, self).__init__()
        self.channels = channels
        self.init_tau = init_tau
        self.register_buffer('eps', torch.Tensor([eps]))
        self.affine = affine

        if self.affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1, channels, 1, 1), requires_grad=True)   # FRN gamma
            self.bias = nn.parameter.Parameter(torch.Tensor(1, channels, 1, 1), requires_grad=True)     # FRN beta
        self.tau = nn.parameter.Parameter(torch.Tensor(1, channels, 1, 1), requires_grad=True)      # TLU: max(x, tau)

        self.initialize_weights()

    def initialize_weights(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        nn.init.constant_(self.tau, self.init_tau)

    def forward(self, x):
        # FRN:
        #   channel-wise mean of L2-norm
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        #   normalize by dividing with the mean L2-norm
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        #   apply scale and bias
        if self.affine:
            x = self.weight * x + self.bias

        # TLU:
        x = torch.max(x, self.tau)
        return x

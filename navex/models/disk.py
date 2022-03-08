import torch
from torch import nn
from torch.nn import functional as F

# install by `pip install git+https://github.com/jatentaki/unets`,
#            `pip install git+https://github.com/jatentaki/torch-localize` and
#            `pip install git+https://github.com/jatentaki/torch-dimcheck`
import unets

from .r2d2 import R2D2
from navex.datasets.tools import unit_aflow


class DISK(R2D2):
    def __init__(self, arch, des_head, det_head, qlt_head, **kwargs):
        if det_head.get('d2d', False):
            det_head['after_des'] = True
            det_head['act_fn'] = 'none'
        super(DISK, self).__init__(arch, des_head, det_head, qlt_head, **kwargs)
        if self.separate_des_head:
            self.des_head = None
            self.det_head = None
            self.qlt_head = None
        if det_head.get('d2d', False):
            self.det_head = D2D()

    def create_backbone(self, arch, cache_dir=None, pretrained=False, width_mult=1.0, in_channels=1,
                        separate_des_head=False, **kwargs):
        bb_ch_out = self.conf['descriptor_dim'] \
                    + ((1 + (0 if self.conf['qlt_head']['skip'] else 1)) if separate_des_head else 0)

        kernel_size = 5
        down_channels = [16, 32, 64, 64, 64]
        up_channels = [64, 64, 64, bb_ch_out]
        setup = {**(unets.fat_setup if arch == 'fat' else unets.thin_setup), 'bias': True, 'padding': True}

        unet = unets.Unet(in_features=in_channels, size=kernel_size, down=down_channels, up=up_channels, setup=setup)

        return unet, up_channels[-1]

    def forward(self, input):
        # input is a pair of images
        features = self.backbone(input)

        if self.separate_des_head:
            dn = self.conf['descriptor_dim']
            des = features[:, :dn, :, :]
            det = features[:, dn: dn+1, :, :]
            qlt = None if self.conf['qlt_head']['skip'] else features[:, -1:, :, :]
        else:
            des = features if getattr(self, 'des_head', None) is None else self.des_head(features)
            des2 = None

            if self.conf['det_head']['after_des']:
                if self.conf['det_head'].get('d2d', False):
                    with torch.no_grad():
                        det = self.det_head(des)
                else:
                    des2 = des.pow(2) if des2 is None else des2
                    det = self.det_head(des2)
            else:
                det = self.det_head(features)

            if self.qlt_head is None:
                qlt = None
            elif self.conf['qlt_head']['after_des']:
                des2 = des.pow(2) if des2 is None else des2
                qlt = self.qlt_head(des2)
            else:
                qlt = self.qlt_head(features)

        output = self.fix_output(des, det, qlt)
        return output


class D2D(nn.Module):
    def __init__(self, k=5, d=4):
        super(D2D, self).__init__()
        self.k = k  # relative salience window diameter
        self.d = d  # relative salience window dilation
        self._byx = None

    def forward(self, des):
        B, D, H, W = des.shape

        s_abs = torch.sqrt(torch.mean(des.pow(2), dim=1, keepdim=True)
                           - torch.mean(des, dim=1, keepdim=True).pow(2))

        # s_rel[i,j] = sum(norm(des[i, j] - des[i+u, j+v]), uv=-k/2*d:k/2*d:d)
        if self._byx is None or self._byx[0].shape != (B, self.k ** 2, H, W):
            xy = torch.LongTensor(unit_aflow(W, H)).to(des.device).view(-1, 2).t()
            offsets = list(range(-(self.k//2)*self.d, self.k*(self.d//2) + 1, self.d))
            offsets = torch.LongTensor([(i, j) for i in offsets
                                               for j in offsets]).to(des.device).view(-1, 2).t()

            xy = xy[:, None, :] + offsets[:, :, None]
            torch.clamp(xy[0], 0, W - 1, out=xy[0])
            torch.clamp(xy[1], 0, H - 1, out=xy[1])

            b = torch.arange(0, B, device=des.device).view(B, 1, 1, 1).expand(B, self.k ** 2, H, W)
            xy = xy.view(1, 2, self.k ** 2, H, W).expand(B, 2, self.k ** 2, H, W)
            self._byx = (b,  xy[:, 1, :, :, :], xy[:, 0, :, :, :])

        des_p = des.permute(0, 2, 3, 1)
        des_a = des_p[self._byx[0], self._byx[1], self._byx[2], :]
        des_n = torch.linalg.norm(des_p.view(B, 1, H, W, D).expand(B, self.k**2, H, W, D) - des_a, dim=4)
        s_rel = torch.mean(des_n, dim=1).view(B, 1, H, W)
        s_raw = s_abs * s_rel

        # always positive, however, need to scale to range 0 - 1
        s = s_raw / (s_raw + 1)
        return s

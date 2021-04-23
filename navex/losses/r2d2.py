import math

import torch
from r2d2.nets.sampler import NghSampler2
from torch import nn
from torch.functional import F
from torch.nn import BCELoss

from .base import BaseLoss
from .ap import AveragePrecisionLoss, DiscountedAPLoss
from .cosim import CosSimilarityLoss
from .peakiness import PeakinessLoss


class R2D2Loss(BaseLoss):
    def __init__(self, wdt=1.0, wap=1.0, wqt=1.0, det_n=16, base=0.5, nq=20, sampler=None):
        super(R2D2Loss, self).__init__()

        self.wdt = -math.log(wdt) if wdt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wdt)]))
        self.wpk = -math.log(wdt) if wdt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wdt)]))
        self.wap = -math.log(wap) if wap >= 0 else nn.Parameter(torch.Tensor([-math.log(-wap)]))
        self.wqt = -math.log(wqt) if wqt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wqt)]))

        self.cosim_loss = CosSimilarityLoss(int(det_n))
        self.peakiness_loss = PeakinessLoss(int(det_n))
        self.ap_loss = DiscountedAPLoss(base=base, nq=nq, sampler_conf=sampler)
#        self.ap_loss = AveragePrecisionLoss(base=base, nq=nq, sampler_conf=sampler)

    @property
    def base(self):
        return self.ap_loss.base

    @property
    def border(self):
        return self.ap_loss.sampler.border

    def update_conf(self, new_conf):
        ok = True
        for k, v in new_conf.items():
            if k in ('wdt', 'wap'):
                ov = getattr(self, k)
                if isinstance(ov, nn.Parameter):
                    setattr(self, k,  nn.Parameter(torch.Tensor([abs(v)], device=ov.device)))
                else:
                    setattr(self, k, abs(v))
            elif k == 'base':
                self.ap_loss.base = v
            elif k == 'det_n':
                assert v % 2 == 0, 'N must be pair'
                self.cosim_loss.super.name = f'cosim{v}'
                self.cosim_loss.super.patches = nn.Unfold(v, padding=0, stride=v//2)
                self.peakiness_loss.super.name = f'peaky{v}'
                self.peakiness_loss.super.preproc = nn.AvgPool2d(3, stride=1, padding=1)
                self.peakiness_loss.super.maxpool = nn.MaxPool2d(v + 1, stride=1, padding=v // 2)
                self.peakiness_loss.super.avgpool = nn.AvgPool2d(v + 1, stride=1, padding=v // 2)
            else:
                ok = False
        return ok

    def forward(self, output1, output2, aflow):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2

        p_loss = self.peakiness_loss(det1, det2)
        c_loss = self.cosim_loss(det1, det2, aflow)

        # downscale aflow to des and qlt shape
        th, tw = des1.shape[2:]
        sh, sw = aflow.shape[2:]
        if (sh, sh) != (th, th):
            sc = min(th/sh, tw/sw)
            sc_aflow = F.interpolate(aflow, size=(th, tw), mode='nearest') * sc
        else:
            sc_aflow = aflow

        a_loss, q_loss = self.ap_loss(des1, des2, qlt1, qlt2, sc_aflow)

        # maybe optimize weights during training, see
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        #  => regression: -0.5*log(2*w); classification: -0.5*log(w)
        # however, in the papar log(sigma**2) is optimized instead

        eps = 1e-5
        lib = math if isinstance(self.wdt, float) else torch
        p_loss, c_loss = -lib.log(1 - p_loss + eps), -lib.log(1 - c_loss + eps)
        p_loss = lib.exp(-self.wpk) * p_loss + 0.5 * self.wpk
        c_loss = lib.exp(-self.wdt) * c_loss + 0.5 * self.wdt

        lib = math if isinstance(self.wap, float) else torch
        a_loss = lib.exp(-self.wap) * a_loss + 0.5 * self.wap

        lib = math if isinstance(self.wqt, float) else torch
        q_loss = lib.exp(-self.wqt) * q_loss + 0.5 * self.wqt

        # p_loss = self.wdt*p_loss - 0.5*(math.log(self.wdt) if isinstance(self.wdt, float) else torch.log(self.wdt))
        # c_loss = self.wdt*c_loss - 0.5*(math.log(self.wdt) if isinstance(self.wdt, float) else torch.log(self.wdt))
        # a_loss = self.wap*a_loss - 0.5*(math.log(self.wap) if isinstance(self.wap, float) else torch.log(self.wap))

        return p_loss + c_loss + a_loss + q_loss

    def params_to_optimize(self, split=False):
        params = [v for n, v in self.named_parameters() if n in ('wdt', 'wpk', 'wap', 'wqt')]
        if split:
            # new_biases, new_weights, biases, weights, others
            return [[], [], [], [], params]
        else:
            return params

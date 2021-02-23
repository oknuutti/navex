import math

import torch
from torch import nn
from torch.functional import F

from .base import BaseLoss
from .ap import AveragePrecisionLoss
from .cosim import CosSimilarityLoss
from .peakiness import PeakinessLoss


class R2D2Loss(BaseLoss):
    def __init__(self, wdt=1.0, wap=1.0, det_n=16, base=0.5, nq=20, sampler=None):
        super(R2D2Loss, self).__init__()

        self.wdt = wdt if wdt >= 0 else nn.Parameter(torch.Tensor([-wdt]))
        self.wap = wap if wap >= 0 else nn.Parameter(torch.Tensor([-wap]))

        self.ap_loss = AveragePrecisionLoss(base=base, nq=nq, sampler_conf=sampler)
        self.cosim_loss = CosSimilarityLoss(int(det_n))
        self.peakiness_loss = PeakinessLoss(int(det_n))

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
                self.ap_loss.super.base = v
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

        a_loss = self.ap_loss(des1, des2, qlt1, qlt2, sc_aflow)

        # maybe optimize weights during training, see
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        #  => regression: -0.5*log(2*w); classification: -0.5*log(w)
        p_loss = self.wdt*p_loss - 0.5*(math.log(self.wdt) if isinstance(self.wdt, float) else torch.log(self.wdt))
        c_loss = self.wdt*c_loss - 0.5*(math.log(self.wdt) if isinstance(self.wdt, float) else torch.log(self.wdt))
        a_loss = self.wap*a_loss - 0.5*(math.log(self.wap) if isinstance(self.wap, float) else torch.log(self.wap))

        return p_loss + c_loss + a_loss

    def params_to_optimize(self, split=False):
        params = []
        if not isinstance(self.wdt, float):
            params.append(self.wdt)
        if not isinstance(self.wap, float):
            params.append(self.wap)

        if split:
            # new_biases, new_weights, biases, weights, others
            return [[], [], [], [], params]
        else:
            return params

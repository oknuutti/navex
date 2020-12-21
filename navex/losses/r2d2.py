import math

import torch
from torch import nn
from torch.functional import F

from .base import BaseLoss
from .ap import AveragePrecisionLoss
from .cosim import CosSimilarityLoss
from .peakiness import PeakinessLoss


class R2D2Loss(BaseLoss):
    def __init__(self, wc=1.0, wdt=1.0, wap=1.0, det_n=16, base=0.5, nq=20, sampler=None):
        super(R2D2Loss, self).__init__()

        # self.wdt = wdt if wdt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wdt)]))
        # self.wap = wap if wap >= 0 else nn.Parameter(torch.Tensor([-math.log(-wap)]))
        self.wc = wc if wc >= 0 else nn.Parameter(torch.Tensor([-wc]))
        self.wdt = wdt if wdt >= 0 else nn.Parameter(torch.Tensor([-wdt]))
        self.wap = wap if wap >= 0 else nn.Parameter(torch.Tensor([-wap]))

        self.ap_loss = AveragePrecisionLoss(base=base, nq=nq, sampler_conf=sampler)
        self.cosim_loss = CosSimilarityLoss(det_n)
        self.peakiness_loss = PeakinessLoss(det_n)

    def forward(self, output1, output2, aflow):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2

        p_loss = self.peakiness_loss(det1, det2) + 1e-8
        c_loss = self.cosim_loss(det1, det2, aflow) + 1e-8

        # downscale aflow to des and qlt shape
        th, tw = des1.shape[2:]
        sh, sw = aflow.shape[2:]
        if (sh, sh) != (th, th):
            sc = min(th/sh, tw/sw)
            sc_aflow = F.interpolate(aflow, size=(th, tw), mode='bilinear', align_corners=False) * sc
        else:
            sc_aflow = aflow

        a_loss = self.ap_loss(des1, des2, qlt1, qlt2, sc_aflow) + 1e-8

        # maybe optimize weights during training
        # p_loss = (self.wdt * p_loss) if isinstance(self.wdt, float) else (torch.exp(-self.wdt) * p_loss + self.wdt)
        # c_loss = (self.wdt * c_loss) if isinstance(self.wdt, float) else (torch.exp(-self.wdt) * c_loss + self.wdt)
        # a_loss = (self.wap * a_loss) if isinstance(self.wap, float) else (torch.exp(-self.wap) * a_loss + self.wap)
        p_loss = (self.wdt + 1e-8) * torch.log(p_loss) + (0 if isinstance(self.wdt, float) else -torch.log(self.wdt + 1e-8))
        c_loss = (self.wc + 1e-8) * torch.log(c_loss) + (0 if isinstance(self.wc, float) else -torch.log(self.wc + 1e-8))
        a_loss = (self.wap + 1e-8) * torch.log(a_loss) + (0 if isinstance(self.wap, float) else -torch.log(self.wap + 1e-8))

        return p_loss + c_loss + a_loss

    def params_to_optimize(self, split=False):
        params = []
        if not isinstance(self.wc, float):
            params.append(self.wc)
        if not isinstance(self.wdt, float):
            params.append(self.wdt)
        if not isinstance(self.wap, float):
            params.append(self.wap)

        if split:
            # new_biases, new_weights, biases, weights, others
            return [[], [], [], [], params]
        else:
            return params

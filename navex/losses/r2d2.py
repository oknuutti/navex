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
            sc_aflow = F.interpolate(aflow, size=(th, tw), mode='bilinear', align_corners=False) * sc
        else:
            sc_aflow = aflow

        a_loss = self.ap_loss(des1, des2, qlt1, qlt2, sc_aflow)

        # maybe optimize weights during training, see
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        p_loss = self.wdt*p_loss - 0.5*(math.log(2*self.wdt) if isinstance(self.wdt, float) else torch.log(2*self.wdt))
        c_loss = self.wdt*c_loss - 0.5*(math.log(2*self.wdt) if isinstance(self.wdt, float) else torch.log(2*self.wdt))
        a_loss = self.wap*a_loss - 0.5*(math.log(2*self.wap) if isinstance(self.wap, float) else torch.log(2*self.wap))

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

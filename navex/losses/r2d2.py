
import torch
from torch.functional import F

from .base import BaseLoss
from .ap import AveragePrecisionLoss
from .cosim import CosSimilarityLoss
from .peakiness import PeakinessLoss


class R2D2Loss(BaseLoss):
    def __init__(self, wp=1.0, wc=1.0, wa=1.0, det_n=16, base=0.5, nq=20, sampler=None):
        super(R2D2Loss, self).__init__()

        self.wp = wp
        self.wc = wc
        self.wa = wa

        self.ap_loss = AveragePrecisionLoss(base=base, nq=nq, sampler_conf=sampler)
        self.cosim_loss = CosSimilarityLoss(det_n)
        self.peakiness_loss = PeakinessLoss(det_n)

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

        return self.wp * p_loss + self.wc * c_loss + self.wa * a_loss

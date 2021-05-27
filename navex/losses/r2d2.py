import math

import torch
from torch import nn
from torch.functional import F

from .base import BaseLoss
from .ap import DiscountedAPLoss, WeightedAPLoss, ThresholdedAPLoss
from .cosim import CosSimilarityLoss
from .peakiness import PeakinessLoss


class R2D2Loss(BaseLoss):
    def __init__(self, loss_type, wpk=0.5, wdt=1.0, wap=1.0, wqt=1.0, det_n=16, base=0.5, nq=20, sampler=None):
        super(R2D2Loss, self).__init__()
        self.loss_type = loss_type

        self.cosim_loss = CosSimilarityLoss(int(det_n))
        self.peakiness_loss = PeakinessLoss(int(det_n))

        self.wpk = wpk
        self.wdt = -math.log(wdt) if wdt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wdt)]))
        self.wap = -math.log(wap) if wap >= 0 else nn.Parameter(torch.Tensor([-math.log(-wap)]))
        self.wqt = 0.0

        if loss_type == 'discounted':
            self.wqt = -math.log(wqt) if wqt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wqt)]))
            self.ap_loss = DiscountedAPLoss(base=base, nq=nq, sampler_conf=sampler)
        elif loss_type == 'weighted':
            self.ap_loss = WeightedAPLoss(base=base, nq=nq, sampler_conf=sampler)
        elif loss_type == 'thresholded':
            self.ap_loss = ThresholdedAPLoss(base=base, nq=nq, sampler_conf=sampler)
        else:
            assert False, 'invalid loss_type: %s' % loss_type

    @property
    def base(self):
        return self.ap_loss.base

    @property
    def border(self):
        return self.ap_loss.sampler.border

    @property
    def ap_base(self):
        return self.ap_loss.ap_base

    def update_ap_base(self, map):
        assert isinstance(self.ap_loss, ThresholdedAPLoss), "only valid for loss_type='thresholded'"
        self.ap_loss.update_ap_base(map)

    def update_conf(self, new_conf):
        ok = True
        for k, v in new_conf.items():
            if k in ('wdt', 'wap', 'wqt'):
                ov = getattr(self, k)
                if isinstance(ov, nn.Parameter):
                    setattr(self, k,  nn.Parameter(torch.Tensor([abs(v)]).to(ov.device)))
                else:
                    setattr(self, k, abs(v))
            elif k == 'base':
                if isinstance(self.ap_loss.base, nn.Parameter):
                    self.ap_loss.base[0] = v
                else:
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

    def forward(self, output1, output2, aflow, component_loss=False):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2

        p_loss = 2 * self.wpk * self.peakiness_loss(det1, det2)
        c_loss = 2 * (1 - self.wpk) * self.cosim_loss(det1, det2, aflow)

        # downscale aflow to des and qlt shape
        th, tw = des1.shape[2:]
        sh, sw = aflow.shape[2:]
        if (sh, sh) != (th, th):
            sc = min(th/sh, tw/sw)
            sc_aflow = F.interpolate(aflow, size=(th, tw), mode='nearest') * sc
        else:
            sc_aflow = aflow

        a_loss, q_loss = self.ap_loss(output1, output2, sc_aflow)

        # maybe optimize weights during training, see
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        #  => regression: -0.5*log(2*w); classification: -0.5*log(w)
        # log(sigma**2) is optimized

        eps = 1e-5
        lib = math if isinstance(self.wdt, float) else torch
        p_loss, c_loss = -lib.log(1 - p_loss + eps), -lib.log(1 - c_loss + eps)
        p_loss = lib.exp(-self.wdt) * p_loss + 0.5 * self.wdt
        c_loss = lib.exp(-self.wdt) * c_loss + 0.5 * self.wdt

        lib = math if isinstance(self.wap, float) else torch
        a_loss = lib.exp(-self.wap) * a_loss + 0.5 * self.wap

        if self.loss_type == 'weighted':
            a_loss = a_loss + q_loss
            q_loss = None

        if q_loss is not None:
            lib = math if isinstance(self.wqt, float) else torch
            q_loss = lib.exp(-self.wqt) * q_loss + 0.5 * self.wqt
        else:
            q_loss = torch.Tensor([0]).to(des1.device)

        loss = torch.stack((p_loss, c_loss, a_loss, q_loss), dim=1)
        return loss if component_loss else loss.sum(dim=1)

    def params_to_optimize(self, split=False):
        params = [v for n, v in self.named_parameters() if n in ('wdt', 'wap', 'wqt')]
        if split:
            # new_biases, new_weights, biases, weights, others
            return [[], [], [], [], params]
        else:
            return params

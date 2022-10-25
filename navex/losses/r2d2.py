import math

import torch
from torch import nn
from torch.functional import F

from .base import BaseLoss
from .ap import DiscountedAPLoss, WeightedAPLoss, ThresholdedAPLoss, LogThresholdedAPLoss
from .cosim import CosSimilarityLoss
from .disk import DiskLoss
from .peakiness import PeakinessLoss, ActivationLoss


class R2D2Loss(BaseLoss):
    def __init__(self, loss_type, wpk=0.5, wdt=1.0, wap=1.0, wqt=1.0, det_n=16, base=0.5, nq=20, sampler=None):
        super(R2D2Loss, self).__init__()
        self.loss_type = loss_type

        self.cosim_loss = CosSimilarityLoss(int(det_n), use_max=wpk > 0)
        if wpk > 0:
            self.peakiness_loss = ActivationLoss()
            self._wpk = wpk
        else:
            self.peakiness_loss = PeakinessLoss(int(det_n))
            self._wpk = abs(wpk)

        self.wdt = None if wdt == 0 else (-math.log(wdt) if wdt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wdt)])))
        self.wap = -math.log(wap) if wap >= 0 else nn.Parameter(torch.Tensor([-math.log(-wap)]))
        self.wqt = 0.0

        if loss_type == 'discounted':
            self.wqt = -math.log(wqt) if wqt >= 0 else nn.Parameter(torch.Tensor([-math.log(-wqt)]))
            self.ap_loss = DiscountedAPLoss(base=base, nq=nq, sampler_conf=sampler)
        elif loss_type == 'weighted':
            self.ap_loss = WeightedAPLoss(base=base, nq=nq, sampler_conf=sampler)
        elif loss_type == 'thresholded':
            self.ap_loss = ThresholdedAPLoss(base=base, nq=nq, warmup_batches=1500, sampler_conf=sampler)
            self.log_peakiness = wqt < 0
        elif loss_type == 'logthresholded':
            self.ap_loss = LogThresholdedAPLoss(base=base, nq=nq, sampler_conf=sampler)
        elif loss_type in ('disk', 'disk-p'):
            self.wdt = None
            self.ap_loss = DiskLoss(reward=abs(wqt), penalty=abs(wdt), sampling_cost=abs(wpk),
                                    sampling_cost_coef=sampler['subd'], cell_d=int(det_n), match_theta=base,
                                    warmup_batches=nq, sampler=sampler, prob_input=loss_type == 'disk-p')
        else:
            assert False, 'invalid loss_type: %s' % loss_type

    @property
    def wpk(self):
        return self.ap_loss.sampling_cost if isinstance(self.ap_loss, DiskLoss) else self._wpk

    @wpk.setter
    def wpk(self, wpk):
        if isinstance(self.ap_loss, DiskLoss):
            self.ap_loss.sampling_cost = wpk
        else:
            self._wpk = wpk

    @property
    def base(self):
        return self.ap_loss.match_theta if isinstance(self.ap_loss, DiskLoss) else self.ap_loss.base

    @base.setter
    def base(self, base):
        if isinstance(self.ap_loss, DiskLoss):
            self.ap_loss.match_theta = base
        else:
            self.ap_loss.base = base

    @property
    def border(self):
        return self.ap_loss.sampler.border

    @property
    def ap_base(self):
        return self.ap_loss.ap_base

    def batch_end_update(self, accs):
        for loss_fn in (self.ap_loss, self.peakiness_loss, self.cosim_loss):
            if hasattr(loss_fn, 'batch_end_update'):
                loss_fn.batch_end_update(accs)

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
                self.cosim_loss.patches = nn.Unfold(v, padding=0, stride=v // 2)
                self.peakiness_loss.max_pool_n = nn.MaxPool2d(v + 1, stride=1, padding=v // 2)
                self.peakiness_loss.avg_pool_n = nn.AvgPool2d(v + 1, stride=1, padding=v // 2)
            else:
                ok = False
        return ok

    def forward(self, output1, output2, aflow, component_loss=False):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2

        # downscale aflow to des and qlt shape
        th, tw = des1.shape[2:]
        sh, sw = aflow.shape[2:]
        if (sh, sh) != (th, th):
            sc = min(th/sh, tw/sw)
            sc_aflow = F.interpolate(aflow, size=(th, tw), mode='nearest') * sc
        else:
            sc_aflow = aflow

        if isinstance(self.ap_loss, DiskLoss):
            tmp = self.ap_loss(output1, output2, sc_aflow, component_loss=True)
            a_loss, q_loss = tmp[0, 2:3], [tmp[0, 3:4]]
        else:
            a_loss, *q_loss = self.ap_loss(output1, output2, sc_aflow)
        q_loss = None if len(q_loss) == 0 else q_loss[0]

        # maybe optimize weights during training, see
        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        #  => regression: -0.5*log(2*w); classification: -0.5*log(w)
        # log(sigma**2) is optimized

        if self.wdt is not None and (det1.requires_grad or det2.requires_grad):
            p_loss = self.peakiness_loss(det1, det2)
            c_loss = self.cosim_loss(det1, det2, aflow)

            eps = 1e-5
            lib = math if isinstance(self.wdt, float) else torch
            if self.log_peakiness:
                p_loss = -torch.log(self.peakiness_loss.max_loss - p_loss + eps) * self.wpk * 2
                c_loss = -torch.log(1 - c_loss + eps) * (1 - self.wpk) * 2
            else:
                p_loss = p_loss * self.wpk * 2
                c_loss = c_loss * (1 - self.wpk) * 2
            p_loss = lib.exp(-self.wdt) * p_loss + 0.5 * self.wdt
            c_loss = lib.exp(-self.wdt) * c_loss + 0.5 * self.wdt
        else:
            p_loss, c_loss = torch.Tensor([0]).to(des1.device), torch.Tensor([0]).to(des1.device)

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

        p_loss, c_loss, a_loss, q_loss = map(torch.atleast_1d, (p_loss, c_loss, a_loss, q_loss))
        loss = torch.stack((p_loss, c_loss, a_loss, q_loss), dim=1)
        return loss if component_loss else loss.sum(dim=1)

    def params_to_optimize(self, split=False):
        params = [v for n, v in self.named_parameters() if n in ('wdt', 'wap', 'wqt', 'ap_loss.match_theta')]
        if split:
            # new_biases, new_weights, biases, weights, others
            return [[], [], [], [], params]
        else:
            return params

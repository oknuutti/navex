import math

import torch
from r2d2.nets.ap_loss import APLoss
from torch.nn import Module, BCELoss

from r2d2.nets.reliability_loss import ReliabilityLoss, PixelAPLoss
from r2d2.nets.sampler import NghSampler2


class AveragePrecisionLoss(Module):
    def __init__(self, base=0.5, nq=20, sampler_conf=None):
        super(AveragePrecisionLoss, self).__init__()
        sampler_conf = sampler_conf or {'ngh': 7, 'subq': -8, 'subd': 1, 'pos_d': 3, 'neg_d': 5, 'border': 16,
                                        'subd_neg': -8, 'maxpool_pos': True}
#        self.super = ReliabilityLoss(sampler=NghSampler2(**sampler_conf), base=base, nq=nq)
        self.super = WeightedAPLoss(sampler=NghSampler2(**sampler_conf), nq=nq)

    def forward(self, des1, des2, qlt1, qlt2, aflow):
        assert des1.shape == des2.shape, 'different shape descriptor tensors'
        assert qlt1.shape == qlt2.shape, 'different shape quality tensors'
        assert des1.shape[2:] == qlt2.shape[2:], 'different shape descriptor and quality tensors'
        assert des1.shape[2:] == aflow.shape[2:], 'different shape absolute flow tensor'
        return self.super((des1, des2), aflow, reliability=(qlt1, qlt2))


class WeightedAPLoss(PixelAPLoss):
    """
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    used as inspiration
    """
    def __init__(self, sampler, nq=20):
        PixelAPLoss.__init__(self, sampler, nq=nq)
        self.base = 0.0     # TODO: remove
        self.eps = 1e-5
        self.max_rel = -math.log(self.eps)
        self.name = 'reliability'

    def loss_from_ap(self, ap, rel):
        # rel ~ log(1/sigma**2), i.e. log precision
        rel_capped = rel.clamp(-self.max_rel, self.max_rel)
        return - 0.5 * torch.exp(rel_capped) * torch.log(ap + self.eps) - 0.5 * rel_capped


class DiscountedAPLoss(Module):
    def __init__(self, base=0.5, scale=10, nq=20, min=0, max=1, euc=False, sampler_conf=None):
        super(DiscountedAPLoss, self).__init__()

        self.base = base
        self.scale = scale
        self.name = 'reliability'

        self.sampler = NghSampler2(**(sampler_conf or {'ngh': 7, 'subq': -8, 'subd': 1, 'pos_d': 3, 'neg_d': 5,
                                                       'border': 16, 'subd_neg': -8, 'maxpool_pos': True}))
        self.calc_ap = APLoss(nq=nq, min=min, max=max, euc=euc)
        self.bce_loss = BCELoss(reduction='none')

    def forward(self, des1, des2, qlt1, qlt2, aflow):
        # subsample things
        scores, gt, mask, qqlt = self.sampler((des1, des2), (qlt1, qlt2), aflow)

        n = qqlt.numel()
        scores, gt, qqlt = scores.view(n, -1), gt.view(n, -1), qqlt.view(n, -1)
        ap = self.calc_ap(scores, gt).view(n, -1)

        tmp = 1 - ap
        a_loss = self.scale * torch.log(torch.exp((- tmp + self.base) / self.scale) + 1) + tmp
        a_loss = a_loss.view(mask.shape)[mask].mean()

        q_loss = self.bce_loss(qqlt, ap.detach())
        q_loss = q_loss.view(mask.shape)[mask].mean()

        return a_loss, q_loss

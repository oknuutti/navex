import math

import torch
from torch.nn import Module, BCELoss
import torch.nn.functional as F

from r2d2.nets.sampler import NghSampler2
from r2d2.nets.ap_loss import APLoss


class DiscountedAPLoss(Module):
    def __init__(self, base=0.5, scale=0.1, nq=20, min=0, max=1, euc=False, sampler_conf=None):
        super(DiscountedAPLoss, self).__init__()

        self.eps = 1e-5
        self.base = base
        self.scale = scale
        self.bias = self.scale * math.log(math.exp((1 - self.base) / self.scale) + 1)
        self.name = 'reliability'
        self.discount = True

        self.sampler = NghSampler2(**(sampler_conf or {'ngh': 7, 'subq': -8, 'subd': 1, 'pos_d': 3, 'neg_d': 5,
                                                       'border': 16, 'subd_neg': -8, 'maxpool_pos': True}))
        self.calc_ap = APLoss(nq=nq, min=min, max=max, euc=euc)
        self.bce_loss = BCELoss(reduction='none')

    def forward(self, des1, des2, qlt1, qlt2, aflow):
        # subsample things
        scores, gt, mask, rel = self.sampler((des1, des2), (qlt1, qlt2), aflow)

        n = rel.numel()
        scores, gt, rel = scores.view(n, -1), gt.view(n, -1), rel.view(n, -1)
        ap = self.calc_ap(scores, gt).view(n, -1)

        a_loss, q_loss = self.losses(ap, rel)

        a_loss = a_loss.view(mask.shape)[mask].mean()
        q_loss = q_loss.view(mask.shape)[mask].mean() if q_loss is not None else None
        return a_loss, q_loss

    def losses(self, ap, rel):
        # reversed logistic function shaped derivative for loss (x = 1 - ap), arrived at by integration:
        #   integrate(1 - 1/(1+exp(-(x - bias) / scale)), x) => -scale * log(1 + exp(-(x - bias) / scale))
        if self.discount:
            x = 1 - ap
            # a_loss = self.bias - self.scale * torch.log(1 + torch.exp(-(x - (1 - self.base)) / self.scale))
            a_loss = self.bias - F.softplus(-(x - (1 - self.base)), 1 / self.scale)
        else:
            a_loss = -torch.log(ap + self.eps)

        q_loss = self.bce_loss(rel, ap.detach())

        return a_loss, q_loss


class WeightedAPLoss(DiscountedAPLoss):
    """
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    used as inspiration
    """
    def losses(self, ap, rel):
        # rel ~ log(1/sigma**2), i.e. log precision
        # rel_capped = rel.clamp(-self.max_rel, self.max_rel)
        rel = rel + self.eps
        a_loss = - rel * torch.log(ap + self.eps)
        q_loss = - 0.5 * torch.log(rel)
        return a_loss, q_loss


class ThresholdedAPLoss(DiscountedAPLoss):
    def __init__(self, *args, **kwargs):
        super(ThresholdedAPLoss, self).__init__(*args, **kwargs)
        self.base = torch.nn.Parameter(torch.Tensor([self.base]), requires_grad=False)

    """
    Original loss
    """
    def losses(self, ap, rel):
        # TODO: try with -torch.log(rel * ap + (1 - rel) * self.base)
        a_loss = 1 - rel * ap - (1 - rel) * self.base
        return a_loss, None

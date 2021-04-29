import math

import torch
from torch.nn import Module, BCELoss
import torch.nn.functional as F

from r2d2.nets.sampler import NghSampler2
from r2d2.nets.ap_loss import APLoss

from navex.losses.sampler import DetectionSampler


class DiscountedAPLoss(Module):
    def __init__(self, base=0.5, scale=0.1, nq=20, min=0, max=1, euc=False, sampler_conf=None):
        super(DiscountedAPLoss, self).__init__()

        self.eps = 1e-5
        self.base = base
        self.scale = scale
        self.bias = self.scale * math.log(math.exp((1 - self.base) / self.scale) + 1)
        self.name = 'ap-loss'
        self.discount = True

        if 0:
            self.sampler = NghSampler2(**sampler_conf)
        else:
            c = sampler_conf
            self.sampler = DetectionSampler(pos_r=c['pos_d'], cell_d=abs(c['subq']),
                                            border=c['border'], max_neg_b=c['max_neg_b'])

        self.calc_ap = APLoss(nq=nq, min=min, max=max, euc=euc)
        self.bce_loss = BCELoss(reduction='none')

    def forward(self, output1, output2, aflow):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2

        # subsample things
        if 0:
            scores, labels, mask, qlt = self.sampler((des1, des2), (qlt1, qlt2), aflow)
        else:
            scores, labels, mask, qlt = self.sampler(output1, output2, aflow)

        n = qlt.numel()
        scores, labels, qlt = scores.view(n, -1), labels.view(n, -1), qlt.view(n, -1)
        ap = self.calc_ap(scores, labels).view(n, -1)

        a_loss, q_loss = self.losses(ap, qlt)

        a_loss = a_loss.view(mask.shape)[mask].mean()
        q_loss = q_loss.view(mask.shape)[mask].mean() if q_loss is not None else None
        return a_loss, q_loss

    def losses(self, ap, qlt):
        # reversed logistic function shaped derivative for loss (x = 1 - ap), arrived at by integration:
        #   integrate(1 - 1/(1+exp(-(x - bias) / scale)), x) => -scale * log(1 + exp(-(x - bias) / scale))
        if self.discount:
            x = 1 - ap
            # a_loss = self.bias - self.scale * torch.log(1 + torch.exp(-(x - (1 - self.base)) / self.scale))
            a_loss = self.bias - F.softplus(-(x - (1 - self.base)), 1 / self.scale)
        else:
            a_loss = -torch.log(ap + self.eps)

        q_loss = self.bce_loss(qlt, ap.detach())

        return a_loss, q_loss


class WeightedAPLoss(DiscountedAPLoss):
    """
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    used as inspiration
    """
    def losses(self, ap, qlt):
        # qlt ~ log(1/sigma**2), i.e. log precision
        # qlt_capped = qlt.clamp(-self.max_qlt, self.max_qlt)
        qlt = qlt + self.eps
        a_loss = - qlt * torch.log(ap + self.eps)
        q_loss = - 0.5 * torch.log(qlt)
        return a_loss, q_loss


class ThresholdedAPLoss(DiscountedAPLoss):
    def __init__(self, *args, **kwargs):
        super(ThresholdedAPLoss, self).__init__(*args, **kwargs)
        self.base = torch.nn.Parameter(torch.Tensor([self.base]), requires_grad=False)

    def losses(self, ap, qlt):
        # TODO: try with -torch.log(qlt * ap + (1 - qlt) * self.base)
        a_loss = 1 - qlt * ap - (1 - qlt) * self.base
        return a_loss, None


# TODO: remove when no need to load models that refer to it
class AveragePrecisionLoss(Module):
    """
    DEPRECATED.
    """
    def __init__(self, base=0.5, nq=20, sampler_conf=None):
        super(AveragePrecisionLoss, self).__init__()
        sampler_conf = sampler_conf or {'ngh': 7, 'subq': -8, 'subd': 1, 'pos_d': 3, 'neg_d': 5, 'border': 16,
                                        'subd_neg': -8, 'maxpool_pos': True}

        from r2d2.nets.reliability_loss import ReliabilityLoss
        self.super = ReliabilityLoss(sampler=NghSampler2(**sampler_conf), base=base, nq=nq)

    def forward(self, output1, output2, aflow):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2

        assert des1.shape == des2.shape, 'different shape descriptor tensors'
        assert qlt1.shape == qlt2.shape, 'different shape quality tensors'
        assert des1.shape[2:] == qlt2.shape[2:], 'different shape descriptor and quality tensors'
        assert des1.shape[2:] == aflow.shape[2:], 'different shape absolute flow tensor'
        return self.super((des1, des2), aflow, reliability=(qlt1, qlt2))

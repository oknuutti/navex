import math

import torch
from torch.nn import Module, BCELoss
import torch.nn.functional as F

from navex.losses.quantizer import Quantizer
from navex.losses.sampler import GuidedSampler


class DiscountedAPLoss(Module):
    def __init__(self, base=0.5, scale=0.1, nq=20, warmup_batches=6000, sampler_conf=None):
        super(DiscountedAPLoss, self).__init__()

        self.eps = 1e-5
        self.base = base
        self.scale = scale
        self.bias = self.scale * math.log(math.exp((1 - self.base) / self.scale) + 1)
        self.name = 'ap-loss'
        self.discount = False
        self.batch_count = torch.nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.warmup_batches = warmup_batches

        # TODO: update config
        c = sampler_conf
        self.sampler = GuidedSampler(pos_r=c['pos_d'], neg_min_r=c['neg_d'], neg_max_r=c['neg_d'] + c['ngh'],
                                     neg_step=c['subd'], cell_d=abs(c['subq']), border=c['border'],
                                     max_neg_b=c['max_neg_b'], random=1.0)

        self.calc_ap = DifferentiableAP(bins=nq, euclidean=False)  # eucl perf worse, maybe due to lower mid ap res
        self.bce_loss = BCELoss(reduction='none')

    def batch_end_update(self, accs):
        self.batch_count += 1

    def forward(self, output1, output2, aflow):
        scores, labels, mask, qlt = self.sampler(output1, output2, aflow)

        n = qlt.numel()
        scores, labels, qlt = scores.view(n, -1), labels.view(n, -1), qlt.view(n, -1)
        ap = self.calc_ap(scores, labels).view(n, -1)

        a_loss, q_loss = self.losses(ap, qlt)

        a_loss = a_loss.view(mask.shape)[mask].mean()
        q_loss = q_loss.view(mask.shape)[mask].mean() if q_loss is not None else None
        return a_loss, q_loss

    def losses(self, ap, qlt):
        if 0:
            # reversed logistic function shaped derivative for loss (x = 1 - ap), arrived at by integration:
            #   integrate(1 - 1/(1+exp(-(x - bias) / scale)), x) => -scale * log(1 + exp(-(x - bias) / scale))
            x = 1 - ap
            # a_loss = self.bias - self.scale * torch.log(1 + torch.exp(-(x - (1 - self.base)) / self.scale))
            a_loss = self.bias - F.softplus(-(x - (1 - self.base)), 1 / self.scale)
        elif self.discount:
            a_loss = (1 - ap) * (qlt.detach() if self.batch_count > self.warmup_batches else 1.0)
        elif 0:
            a_loss = -torch.log(ap + self.eps)
        else:
            a_loss = 1 - ap

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
    def __init__(self, *args, update_coef=0.003, **kwargs):
        super(ThresholdedAPLoss, self).__init__(*args, **kwargs)
        self.current_map = torch.nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.update_coef = update_coef

    def losses(self, ap, qlt):
        # q*(1-a) + (1-q)*(1-b) => q - q*a + 1 - b - q + q*b => 1 - (q*a +b -q*b) => 1 - (q*a + (1-q)*b)
        a_loss = 1 - (qlt * ap + (1 - qlt) * self.current_map * self.base)
        return a_loss, None

    def batch_end_update(self, accs):
        map = accs[3] if 0 else 1.0

        # update after every training batch
        self.current_map.set_(self.update_coef * map + (1 - self.update_coef) * self.current_map)

    @property
    def ap_base(self):
        return self.current_map * self.base


class LogThresholdedAPLoss(ThresholdedAPLoss):
    def losses(self, ap, qlt):
        eps = 1e-5
        # inspired by binary cross-entropy:  -(y*log(p) + (1-y)*log(1-p)), however, if 1-ap=y, 1-y <> base_ap
        #  - if qlt ~ 1, ap needs to be very close to 1
        #  - never good idea for qlt ~ 0,  ...
        if 0:
            a_loss = -(torch.log(1 - qlt + eps) * (1 - ap) + torch.log(qlt + eps) * (1 - self.current_map * self.base))
        elif 1:
            a_loss = -(torch.log(ap + eps) * qlt + torch.log(self.current_map * self.base + eps) * (1 - qlt))
        else:
            a_loss = torch.log(1 - qlt + eps) * torch.log(ap + eps) \
                     + torch.log(qlt + eps) * torch.log(self.current_map * self.base + eps)

        # was first:
        # a_loss = -torch.log(qlt * ap + (1 - qlt) * self.current_map * self.base + eps)
        return a_loss, None


class DifferentiableAP(Module):
    """
    Based on "Descriptors Optimized for Average Precision" by He et al. 2018
    """
    def __init__(self, bins=25, euclidean=True):
        super(DifferentiableAP, self).__init__()
        self.quantizer = Quantizer(bins, min_v=0, max_v=1)  # note that min_v=0 even though scores can go as low as -1
        self.euclidean = euclidean

    def forward(self, score, label):
        if self.euclidean:        # use `1 - euclidean distance` instead of pure inner product
            score = 1 - torch.sqrt(2.0001 - 2 * score)

        # quantize matching scores, e
        binned_s = self.quantizer(score, insert_dim=1)

        # prepare for ap calculation
        samples_per_bin = binned_s.sum(dim=2)
        correct_per_bin = (binned_s * label[:, None, :].float()).sum(dim=2)
        cum_correct = correct_per_bin.cumsum(dim=1)
        cum_precision = cum_correct / (1e-16 + samples_per_bin.cumsum(dim=1))

        # average precision, per query
        ap = (correct_per_bin * cum_precision).sum(dim=1) / cum_correct[:, -1]

        return ap


# TODO: remove when no need to load models that refer to it
# class AveragePrecisionLoss(Module):
#     """
#     DEPRECATED.
#     """
#     def __init__(self, base=0.5, nq=20, sampler_conf=None):
#         super(AveragePrecisionLoss, self).__init__()
#         sampler_conf = sampler_conf or {'ngh': 7, 'subq': -8, 'subd': 1, 'pos_d': 3, 'neg_d': 5, 'border': 16,
#                                         'subd_neg': -8, 'maxpool_pos': True}
#
#         from r2d2.nets.reliability_loss import ReliabilityLoss
#         from r2d2.nets.sampler import NghSampler2
#
#         self.super = ReliabilityLoss(sampler=NghSampler2(**sampler_conf), base=base, nq=nq)
#
#     def forward(self, output1, output2, aflow):
#         des1, det1, qlt1 = output1
#         des2, det2, qlt2 = output2
#
#         assert des1.shape == des2.shape, 'different shape descriptor tensors'
#         assert qlt1.shape == qlt2.shape, 'different shape quality tensors'
#         assert des1.shape[2:] == qlt2.shape[2:], 'different shape descriptor and quality tensors'
#         assert des1.shape[2:] == aflow.shape[2:], 'different shape absolute flow tensor'
#         return self.super((des1, des2), aflow, reliability=(qlt1, qlt2))

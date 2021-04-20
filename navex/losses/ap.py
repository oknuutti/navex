
from torch.nn import Module

from r2d2.nets.reliability_loss import ReliabilityLoss, PixelAPLoss
from r2d2.nets.sampler import NghSampler2


class AveragePrecisionLoss(Module):
    def __init__(self, base=0.5, nq=20, sampler_conf=None):
        super(AveragePrecisionLoss, self).__init__()
        sampler_conf = sampler_conf or {'ngh': 7, 'subq': -8, 'subd': 1, 'pos_d': 3, 'neg_d': 5, 'border': 16,
                                        'subd_neg': -8, 'maxpool_pos': True}
        self.super = ReliabilityLoss(sampler=NghSampler2(**sampler_conf), base=base, nq=nq)

    def forward(self, des1, des2, qlt1, qlt2, aflow):
        assert des1.shape == des2.shape, 'different shape descriptor tensors'
        assert qlt1.shape == qlt2.shape, 'different shape quality tensors'
        assert des1.shape[2:] == qlt2.shape[2:], 'different shape descriptor and quality tensors'
        assert des1.shape[2:] == aflow.shape[2:], 'different shape absolute flow tensor'
        return self.super((des1, des2), aflow, reliability=(qlt1, qlt2))


class ReliabilityLossBCE(PixelAPLoss):
    """
    Use Binary Cross Entropy Loss
    """
    def __init__(self, sampler, base=0.5, nq=20):
        PixelAPLoss.__init__(self, sampler, nq=nq)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'

    def loss_from_ap(self, ap, rel):
        return 1 - (ap * rel + self.base * (1 - rel))

        # (1 - b) + (b - ap) * r

        #
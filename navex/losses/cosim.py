
from torch.nn import Module

from r2d2.nets.repeatability_loss import CosimLoss


class CosSimilarityLoss(Module):
    def __init__(self, N=16):
        super(CosSimilarityLoss, self).__init__()
        self.super = CosimLoss(N)

    def forward(self, det1, det2, aflow):
        return self.super((det1, det2), aflow)


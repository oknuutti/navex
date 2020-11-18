
from torch.nn import Module

from r2d2.nets.repeatability_loss import PeakyLoss


class PeakinessLoss(Module):
    def __init__(self, N=16):
        super(PeakinessLoss, self).__init__()
        self.super = PeakyLoss(N)

    def forward(self, det1, det2):
        return self.super((det1, det2))


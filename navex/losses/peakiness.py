from torch.nn import Module, AvgPool2d, MaxPool2d


class PeakinessLoss(Module):
    def __init__(self, N=16):
        super(PeakinessLoss, self).__init__()
        self.max_pool_n = MaxPool2d(N + 1, stride=1, padding=N // 2)
        self.avg_pool_n = AvgPool2d(N + 1, stride=1, padding=N // 2)

    def forward(self, det1, det2):
        loss1 = 1 - (self.max_pool_n(det1) - self.avg_pool_n(det1)).mean()
        loss2 = 1 - (self.max_pool_n(det2) - self.avg_pool_n(det2)).mean()
        return (loss1 + loss2) / 2


class ActivationLoss(Module):
    def __init__(self, cost):
        super(ActivationLoss, self).__init__()
        self.cost = cost

    def forward(self, det1, det2):
        return self.cost * (det1.mean() + det2.mean()) / 2

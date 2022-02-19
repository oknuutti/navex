from torch.nn import Module, AvgPool2d, MaxPool2d


class PeakinessLoss(Module):
    def __init__(self, n=16):
        super(PeakinessLoss, self).__init__()
        self.max_loss = 1
        self.avg_pool_3 = AvgPool2d(3, stride=1, padding=1)
        self.max_pool_n = MaxPool2d(n + 1, stride=1, padding=n // 2)
        self.avg_pool_n = AvgPool2d(n + 1, stride=1, padding=n // 2)

    def forward(self, det1, det2):
        det1, det2 = map(self.avg_pool_3, (det1, det2))
        loss1 = 1 - (self.max_pool_n(det1) - self.avg_pool_n(det1)).mean()
        loss2 = 1 - (self.max_pool_n(det2) - self.avg_pool_n(det2)).mean()
        return (loss1 + loss2) / 2


class ActivationLoss(Module):
    """ just testing out something... """
    def __init__(self):
        super(ActivationLoss, self).__init__()
        self.max_loss = 1

    def forward(self, det1, det2):
        return (det1.mean() + det2.mean()) / 2

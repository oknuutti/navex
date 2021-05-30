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


class WeightedPeakinessLoss(Module):
    def __init__(self, activation_cost, n):
        super(WeightedPeakinessLoss, self).__init__()
        self.activation_cost = activation_cost
        self.max_pool_n = MaxPool2d(n + 1, stride=1, padding=n // 2)

    def forward(self, det1, det2):
        # TODO: figure out better reward that would lead to more uncertain detections
        reward1, reward2 = map(self.max_pool_n, (det1, det2))
        reward = (reward1.mean() + reward2.mean()) / 2
        loss = self.activation_cost * (det1.mean() + det2.mean()) / 2   # TODO: try higher costs than 1.0
        return 1 + loss - reward


class ActivationLoss(Module):
    def __init__(self, sparsity, n):
        super(ActivationLoss, self).__init__()
        self.sparsity = sparsity
        self.avg_pool_n = AvgPool2d(n + 1, stride=1, padding=n // 2)

    def forward(self, det1, det2):
        mean_activation1, mean_activation2 = map(self.avg_pool_n, (det1 + self.sparsity, det2 + self.sparsity))
        loss = (mean_activation1.pow(2).mean() + mean_activation2.pow(2).mean()) / 2
        return loss

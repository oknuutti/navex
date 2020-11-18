from torch.nn import Module


class BaseLoss(Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def params_to_optimize(self, split=False, excl_batch_norm=True):
        if split:
            return [[], [], [], [], []]
        else:
            return []

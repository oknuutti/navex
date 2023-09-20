from torch.nn import Module


class BaseLoss(Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.loss_type = None

    def update_conf(self, new_conf):
        return False

    def params_to_optimize(self, split=False):
        if split:
            return [[], [], [], [], []]
        else:
            return []

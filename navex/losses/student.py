import math

import torch
from torch import nn
from torch.functional import F
from torch.nn.modules.loss import L1Loss, BCELoss

from .base import BaseLoss


class L2Loss:
    def __call__(self, out, lbl):
        return torch.norm(out - lbl, dim=1).mean()


class StudentLoss(BaseLoss):
    def __init__(self, des_loss='L1', des_w=1.0, det_w=1.0, qlt_w=1.0, interpolation_mode='bicubic'):
        super(StudentLoss, self).__init__()

        self.des_w = des_w if des_w >= 0 else nn.Parameter(torch.Tensor([-des_w]))
        self.det_w = det_w if det_w >= 0 else nn.Parameter(torch.Tensor([-det_w]))
        self.qlt_w = qlt_w if qlt_w >= 0 else nn.Parameter(torch.Tensor([-qlt_w]))
        self.interpolation_mode = interpolation_mode

        assert des_loss in ('L1', 'L2'), 'invalid descriptor loss function %s' % (des_loss,)
        self.des_loss = L1Loss() if des_loss == 'L1' else L2Loss()
        self.det_loss = BCELoss()
        self.qlt_loss = BCELoss()

    def forward(self, output, label):
        loss_fns = (self.des_loss, self.det_loss, self.qlt_loss)
        weights = (self.des_w, self.det_w, self.qlt_w)
        align_corners = None if self.interpolation_mode in ('nearest', 'area') else False

        # upscale to higher resolution
        losses = []
        for out, lbl, weight, loss_fn in zip(output, label, weights, loss_fns):
            h1, w1 = out.shape[2:]
            h2, w2 = lbl.shape[2:]
            if (h1, w1) != (h2, w2):
                if h1 * w1 < h2 * w2:
                    out = F.interpolate(out, size=(h2, w2), mode=self.interpolation_mode, align_corners=align_corners)
                elif h1 * w1 > h2 * w2:
                    lbl = F.interpolate(lbl, size=(h1, w1), mode=self.interpolation_mode, align_corners=align_corners)

            log = math.log if isinstance(weight, float) else torch.log
            loss = weight * loss_fn(out, lbl) - 0.5 * log(2 * weight)
            losses.append(loss)

        return torch.stack(losses).sum()

    def params_to_optimize(self, split=False):
        params = []
        if not isinstance(self.des_w, float):
            params.append(self.des_w)
        if not isinstance(self.det_w, float):
            params.append(self.det_w)
        if not isinstance(self.qlt_w, float):
            params.append(self.qlt_w)

        if split:
            # new_biases, new_weights, biases, weights, others
            return [[], [], [], [], params]
        else:
            return params

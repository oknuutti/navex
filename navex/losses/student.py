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
    def __init__(self, des_loss='L1', des_w=1.0, det_w=1.0, qlt_w=1.0, interpolation_mode='bilinear'):
        super(StudentLoss, self).__init__()

        self.des_w = -math.log(des_w) if des_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-des_w)]))
        self.det_w = -math.log(det_w) if det_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-det_w)]))
        self.qlt_w = -math.log(qlt_w) if qlt_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-qlt_w)]))

        assert interpolation_mode != 'bicubic', "can't use bicubic as results could overshoot, "\
                                                "would need L2-normalization for des, clip(0,1) for det and qlt"
        self.interpolation_mode = interpolation_mode

        assert des_loss in ('L1', 'L2'), 'invalid descriptor loss function %s' % (des_loss,)
        self.des_loss = L1Loss() if des_loss == 'L1' else L2Loss()
        self.det_loss = BCELoss()
        self.qlt_loss = BCELoss()

    def update_conf(self, new_conf):
        ok = True
        for k, v in new_conf.items():
            if k in ('des_w', 'det_w', 'qlt_w'):
                ov = getattr(self, k)
                if isinstance(ov, nn.Parameter):
                    setattr(self, k,  nn.Parameter(torch.Tensor([abs(v)], device=ov.device)))
                else:
                    setattr(self, k, abs(v))
            elif k == 'des_loss':
                self.des_loss = L1Loss() if v == 'L1' else L2Loss()
            else:
                ok = False
        return ok

    def forward(self, output, label):
        loss_fns = (self.des_loss, self.det_loss, self.qlt_loss)
        weights = (self.des_w, self.det_w, self.qlt_w)
        w_coefs = (1.0, 0.5, 0.5)   # 1.0 if regression, 0.5 if classification
        align_corners = None if self.interpolation_mode in ('nearest', 'area') else False

        # upscale to higher resolution (uses a lot of memory though, could downscale but would seem fishy, hmm...)
        losses = []
        for out, lbl, weight, loss_fn, w_coef in zip(output, label, weights, loss_fns, w_coefs):
            h1, w1 = out.shape[2:]
            h2, w2 = lbl.shape[2:]
            if (h1, w1) != (h2, w2):
                if h1 * w1 < h2 * w2:
                    out = F.interpolate(out, size=(h2, w2), mode=self.interpolation_mode, align_corners=align_corners)
                elif h1 * w1 > h2 * w2:
                    lbl = F.interpolate(lbl, size=(h1, w1), mode=self.interpolation_mode, align_corners=align_corners)

            lib = math if isinstance(weight, float) else torch
            loss = lib.exp(-weight) * loss_fn(out, lbl) + w_coef * weight
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

import math

import torch
from torch import nn
from torch.functional import F
from torch.nn import SmoothL1Loss
from torch.nn.modules.loss import L1Loss, BCELoss, MSELoss

from .base import BaseLoss


class L2Loss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, out, lbl):
        loss = torch.norm(out - lbl, dim=1)
        if self.reduction != 'none':
            return getattr(loss, self.reduction)()
        return loss


class StudentLoss(BaseLoss):
    def __init__(self, des_loss='L1', des_w=1.0, det_w=1.0, qlt_w=1.0, interpolation_mode='bilinear'):
        super(StudentLoss, self).__init__()

        self.des_w = -math.log(des_w) if des_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-des_w)]))
        self.det_w = -math.log(det_w) if det_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-det_w)]))
        self.qlt_w = -math.log(qlt_w) if qlt_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-qlt_w)]))

        assert interpolation_mode != 'bicubic', "can't use bicubic as results could overshoot, "\
                                                "would need L2-normalization for des, clip(0,1) for det and qlt"
        self.interpolation_mode = interpolation_mode

        clss = {'L1': L1Loss, 'L2': L2Loss, 'MSE': MSELoss, 'SmoothL1': SmoothL1Loss}
        assert des_loss in clss, 'invalid descriptor loss function %s' % (des_loss,)

        self.des_loss = clss[des_loss](reduction='none')
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
                self.des_loss = L1Loss(reduction='none') if v == 'L1' else L2Loss(reduction='none')
            else:
                ok = False
        return ok

    def forward(self, output, label, component_loss=False):
        weights = (self.des_w, self.det_w, self.qlt_w)
        align_corners = None if self.interpolation_mode in ('nearest', 'area') else False
        d = {}

        # upscale to higher resolution (uses a lot of memory though, could downscale but would seem fishy, hmm...)
        for name, out, lbl, weight in zip(('des', 'det', 'qlt'), output, label, weights):
            h1, w1 = out.shape[2:]
            h2, w2 = lbl.shape[2:]
            if (h1, w1) != (h2, w2):
                if h1 * w1 < h2 * w2:
                    out = F.interpolate(out, size=(h2, w2), mode=self.interpolation_mode, align_corners=align_corners)
                elif h1 * w1 > h2 * w2:
                    lbl = F.interpolate(lbl, size=(h1, w1), mode=self.interpolation_mode, align_corners=align_corners)
            d[name] = (out, lbl, weight)

        out, lbl, weight = d['det']
        lib = math if isinstance(weight, float) else torch
        det_loss = lib.exp(-weight) * self.det_loss(out, lbl) + 0.5 * weight  # 1.0 if regression, 0.5 if classification

        out, lbl, weight = d['qlt']
        lib = math if isinstance(weight, float) else torch
        qlt_loss = lib.exp(-weight) * self.qlt_loss(out, lbl) + 0.5 * weight

        out, lbl, weight = d['des']
        lib = math if isinstance(weight, float) else torch
        des_loss = lib.exp(-weight) * (d['qlt'][1] * self.des_loss(out, lbl)).mean() + 1.0 * weight

        loss = torch.stack((des_loss, det_loss, qlt_loss), dim=1)
        return loss if component_loss else loss.sum(dim=1)

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

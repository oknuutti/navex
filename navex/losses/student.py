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
    (
        MATCH_METHOD_UNSHUFFLE,
        MATCH_METHOD_BILINEAR,
    ) = range(2)

    def __init__(self, des_loss='L1', des_w=1.0, det_w=1.0, qlt_w=1.0, skip_qlt=False,
                 match_method=MATCH_METHOD_BILINEAR):
        super(StudentLoss, self).__init__()
        self.match_method = match_method

        self.des_w = -math.log(des_w) if des_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-des_w)]))
        self.det_w = -math.log(det_w) if det_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-det_w)]))
        if skip_qlt:
            self.qlt_w = 0.0
        else:
            self.qlt_w = -math.log(qlt_w) if qlt_w >= 0 else nn.Parameter(torch.Tensor([-math.log(-qlt_w)]))
        self.skip_qlt = skip_qlt

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
                    setattr(self, k,  nn.Parameter(torch.Tensor([abs(v)]).to(ov.device)))
                else:
                    setattr(self, k, abs(v))
            elif k == 'des_loss':
                self.des_loss = L1Loss(reduction='none') if v == 'L1' else L2Loss(reduction='none')
            else:
                ok = False
        return ok

    def _match_interp(self, out, lbl, upsample):
        h1, w1 = out.shape[2:]
        h2, w2 = lbl.shape[2:]
        if (h1, w1) != (h2, w2):
            if upsample:
                # upsample to higher resolution, uses a lot of memory though
                if h1 * w1 < h2 * w2:
                    out = F.interpolate(out, size=(h2, w2), mode='bilinear', align_corners=True)
                elif h1 * w1 > h2 * w2:
                    lbl = F.interpolate(lbl, size=(h1, w1), mode='bilinear', align_corners=True)
            else:
                # downsample to lower resolution, some trade off from better efficiency?
                if h1 * w1 < h2 * w2:
                    lbl = F.interpolate(lbl, size=(h1, w1), mode='area')
                elif h1 * w1 > h2 * w2:
                    out = F.interpolate(out, size=(h2, w2), mode='area')
        return out, lbl

    def forward(self, output, label, component_loss=False):
        des_x, det_x, qlt_x = output
        des_y, det_y, qlt_y = label

        if self.skip_qlt:
            det_y = det_y * qlt_y     # merge det and qlt labels to be detection target label

        if self.match_method == StudentLoss.MATCH_METHOD_UNSHUFFLE:
            # selects the values of y at the detection peaks of x

            d_det_y = F.pixel_unshuffle(det_y, 8)
            idxs = torch.argmax(d_det_y, dim=1, keepdim=True)
            if 0:
                # pick even the detection scores to avoid unnecessary penalty, leads to poor positive feedback though?
                m_det_x = torch.gather(F.pixel_unshuffle(det_x, 8), 1, idxs)
                m_det_y = torch.gather(d_det_y, 1, idxs)
            else:
                m_det_x, m_det_y = det_x, det_y

            d_qlt_y = F.pixel_unshuffle(qlt_y, 8)
            m_qlt_x, lo_qlt_y = qlt_x, torch.gather(d_qlt_y, 1, idxs)
            m_qlt_y = qlt_y if m_qlt_x.shape == qlt_y.shape else lo_qlt_y

            d_des_y = F.pixel_unshuffle(des_y[:, :, None, :, :], 8)
            dI = idxs[:, None, :, :, :].expand(-1, des_y.size(1), 1, -1, -1)
            m_des_x, m_des_y = des_x, torch.gather(d_des_y, 2, dI)[:, :, 0, :, :]
            md_qlt_y = qlt_y if qlt_y.shape == m_des_y.shape else lo_qlt_y
            assert m_des_x.shape == m_des_y.shape, \
                'trained des output assumed to be (H//8, W//8) of teacher model output'
        else:
            # NOTE: det_x is forced to be peaky using softmax in 8x8 blocks, whereas det_y is not => unnecessary penalty
            assert det_x.shape == det_y.shape, 'should not need to match dimensions of detector output'
            m_des_x, m_des_y = self._match_interp(des_x, des_y, upsample=True)
            md_qlt_y = qlt_y
            if not self.skip_qlt:
                m_qlt_x, m_qlt_y = self._match_interp(qlt_x, qlt_y, upsample=False)
            else:
                m_qlt_x, m_qlt_y = qlt_x, qlt_y
            m_det_x, m_det_y = det_x, det_y

        def multitarget_loss(loss, log_sigma2, is_reg):
            # from "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics"
            # https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
            # exp(-log_sigma2) == exp(-log(sigma**2)) == exp(log(sigma**-2)) == 1/(sigma**2)
            # 0.5 * log_sigma2 == 0.5 * log(sigma**2) == log(sigma)
            # coef: 1/2 if regression, 1 if classification
            coef = 0.5 if is_reg else 1.0
            lib = math if isinstance(log_sigma2, float) else torch
            return torch.atleast_1d(coef * lib.exp(-log_sigma2) * loss + 0.5 * log_sigma2)

        if not self.skip_qlt:
            qlt_loss = multitarget_loss(self.qlt_loss(m_qlt_x, m_qlt_y), self.qlt_w, True)
        else:
            qlt_loss = torch.Tensor([0]).to(des_x.device)

        det_loss = multitarget_loss(self.det_loss(m_det_x, m_det_y), self.det_w, True)
        des_loss = multitarget_loss((md_qlt_y * self.des_loss(m_des_x, m_des_y)).mean(), self.des_w, True)

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

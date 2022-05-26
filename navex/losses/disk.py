import math

import torch
from torch import nn
from torch.functional import F
from torch.distributions import Categorical

from .base import BaseLoss
from .sampler import DetectionSampler


class DiskLoss(BaseLoss):
    def __init__(self, sampling_cost=0.001, cell_d=8, match_theta=50, sampler=None,
                 warmup_batch_scale=500, prob_input=False):
        super(DiskLoss, self).__init__()
        self.sampler = DetectionSampler(cell_d=cell_d, border=sampler['border'], random=1.0, max_b=sampler['max_neg_b'],
                                        prob_input=prob_input, sample_matches=sampler['maxpool_pos'] > 0)
        self.max_px_err = sampler['pos_d']
        self.warmup_batch_scale = warmup_batch_scale
        self.prob_input = prob_input
        self.match_theta = match_theta
        self.reward = -1.0
        self.penalty = 0.25
        self.sampling_cost = sampling_cost

        self.batch_count = torch.nn.Parameter(torch.Tensor([-1]), requires_grad=False)
        self._match_theta = None
        self._reward = None
        self._penalty = None
        self._sampling_cost = None
        self.batch_end_update(None)

    @property
    def border(self):
        return self.sampler.border

    def batch_end_update(self, accs):
        self.batch_count += 1
        e = self.batch_count.item() / self.warmup_batch_scale
        if 0:
            # original schedule
            if e < 250/5000:
                ramp = 0.0
            elif e < 5250/5000:
                ramp = 0.1
            else:
                # 1.0 at e=4.05, if allow float e, would be at 3.55
                ramp = min(1., 0.1 + 0.2 * int(e - 250/5000 + 1))
        else:
            # smoother version of above
            if e < 1.05:
                ramp = max(0, min(1, 0.2 * (e - 0.05)))
            else:
                ramp = min(1, 0.2 + 0.32 * (e - 1.05))  # 1.0 at e=3.55

        self._match_theta = self.match_theta*(15/50) + self.match_theta*(35/50) * min(1., 0.05 * e)  # 1.0 at e=20 (!)
        self._reward = 1.0 * self.reward
        self._penalty = ramp * self.penalty
        self._sampling_cost = ramp * self.sampling_cost

        for loss_fn in (self.sampler, ):
            if hasattr(loss_fn, 'batch_end_update'):
                loss_fn.batch_end_update(accs)

    def params_to_optimize(self, split=False):
#        params = [v for n, v in self.named_parameters() if n in ('wdt', 'wap', 'wqt')]
        params = []
        if split:
            # new_biases, new_weights, biases, weights, others
            return [[], [], [], [], params]
        else:
            return params

    def update_conf(self, new_conf):
        ok = True
        for k, v in new_conf.items():
            if k in ('wdt', 'wap', 'wqt'):
                ov = getattr(self, k)
                if isinstance(ov, nn.Parameter):
                    setattr(self, k,  nn.Parameter(torch.Tensor([abs(v)]).to(ov.device)))
                else:
                    setattr(self, k, abs(v))
            elif k == 'base':
                if isinstance(self.ap_loss.base, nn.Parameter):
                    self.ap_loss.base[0] = v
                else:
                    self.ap_loss.base = v
            elif k == 'det_n':
                assert v % 2 == 0, 'N must be pair'
                self.cosim_loss.patches = nn.Unfold(v, padding=0, stride=v // 2)
                self.peakiness_loss.max_pool_n = nn.MaxPool2d(v + 1, stride=1, padding=v // 2)
                self.peakiness_loss.avg_pool_n = nn.AvgPool2d(v + 1, stride=1, padding=v // 2)
            else:
                ok = False
        return ok

    def forward(self, output1, output2, aflow, component_loss=False):
        # based on the article https://proceedings.neurips.cc/paper/2020/file/a42a596fc71e17828440030074d15e74-Paper.pdf
        # and the implementation at https://github.com/cvlab-epfl/disk/blob/master/disk/loss/reinforce.py

        det_logp_mxs, des_dist_mxs, px_dist_mxs, masks, b1s, b2s, sample_logp = self.sampler(output1, output2, aflow)

        if 1:
            q_loss = self._sampling_cost * sample_logp
        else:
            # try a more direct approch to penalizing activation
            (_, det1, *_), (_, det2, *_) = output1, output2
            q_loss = self._sampling_cost * (torch.sigmoid(det1).sum() + torch.sigmoid(det2).sum())

        a_loss = 0
        for det_logp_mx, des_dist_mx, px_dist_mx, mask, b1, b2 in zip(det_logp_mxs, des_dist_mxs, px_dist_mxs, masks, b1s, b2s):
            mask = torch.logical_not(mask)
            n, m = des_dist_mx.shape

            cost_mx = des_dist_mx.new_ones(des_dist_mx.shape, dtype=des_dist_mx.dtype) * self._penalty
            same_b = b1.view(n, 1).expand(n, m) == b2.view(1, m).expand(n, m)
            cost_mx[torch.logical_and(px_dist_mx < self.max_px_err, same_b)] = self._reward
            cost_mx[mask.view((n, 1)).expand((n, m))] = 0

            if 0:
                des_logp12 = Categorical(logits=-self._match_theta * des_dist_mx).logits
                des_logp21 = Categorical(logits=-self._match_theta * des_dist_mx.t()).logits.t()
            else:
                # same as above
                des_logp12 = F.log_softmax(-self._match_theta * des_dist_mx, dim=1)
                des_logp21 = F.log_softmax(-self._match_theta * des_dist_mx, dim=0)

            des_logp_mx = des_logp12 + des_logp21
            with torch.no_grad():
                des_p_mx = torch.exp(des_logp_mx)

            # notice that des_p_mx needs to be detached
            sample_plogp = des_p_mx * (des_logp_mx + det_logp_mx)
            a_loss = a_loss + (cost_mx * sample_plogp).sum()

        dummy = torch.Tensor([0.0]).to(a_loss.device)
        p_loss, c_loss, a_loss, q_loss = map(torch.atleast_1d, (dummy, dummy, a_loss, q_loss))
        loss = torch.stack((p_loss, c_loss, a_loss, q_loss), dim=1)
        return loss if component_loss else loss.sum(dim=1)

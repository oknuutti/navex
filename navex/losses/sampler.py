import math

import numpy as np

import torch
from torch.distributions import Categorical, Bernoulli
import torch.nn.functional as F

from navex.datasets.tools import unit_aflow


class GuidedSampler(torch.nn.Module):
    """
    Sample a detection map by dividing it to certain size cells and selecting a pixel from each based on its
    detection score. Use aflow to match each selected pixel with the corresponding ones in the second image.
    These form the set of positive samples. Negative samples are generated from the neighborhood of the selected pixels
    and extra negative samples are generated from other images in the same batch.
    """

    def __init__(self, pos_r=3, neg_min_r=7, neg_max_r=8, neg_step=2, cell_d=8, border=16, max_neg_b=8, random=False):
        """
        :param pos_r: radius from inside which positive samples are gathered
        :param neg_min_r: min radius for negative samples
        :param neg_max_r: max radius for negative samples
        :param neg_step: negative sampling step
        :param cell_d: diameter of rectangular cell that is searched for max detection score
        :param border: border width, don't sample if closer than this to image borders
        :param max_neg_b: get distractors from at most this amount of images in the mini-batch
        :param random: if True, sample randomly instead of using the detection scores
        """
        super(GuidedSampler, self).__init__()
        self.pos_r = pos_r
        self.neg_min_r = neg_min_r
        self.neg_max_r = neg_max_r
        self.neg_step = neg_step
        self.max_neg_b = max_neg_b
        self.random_sampler = WeightedRandomSampler(cell_d, border, random, subsample=False, act_logp=False)

        r = int(self.pos_r)
        pos_offsets = [
            (i, j)
            for i in range(-r, r + 1)
            for j in range(-r, r + 1)
            if (i**2 + j**2) <= self.pos_r**2
        ]
        self.pos_offsets = torch.nn.Parameter(torch.LongTensor(pos_offsets).view(-1, 2).t(), requires_grad=False)

        self.neg_offsets = None
        if self.neg_min_r < self.neg_max_r:
            neg_offsets = [
                (i, j)
                for i in range(-self.neg_max_r, self.neg_max_r + 1, self.neg_step)
                for j in range(-self.neg_max_r, self.neg_max_r + 1, self.neg_step)
                if self.neg_min_r**2 <= (i**2 + j**2) <= self.neg_max_r**2
            ]
            self.neg_offsets = torch.nn.Parameter(torch.LongTensor(neg_offsets).view(-1, 2).t(), requires_grad=False)

    @property
    def border(self):
        return self.random_sampler.border

    @border.setter
    def border(self, border):
        self.random_sampler.border = border

    def forward(self, output1, output2, aflow):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2
        B, _, H, W = aflow.shape

        b, y1, x1, _ = self.random_sampler(det1)
        s_des1 = des1[b, :, y1, x1]

        xy2 = (aflow[b, :, y1, x1] + 0.5).long().t()
        mask = ((0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)).view(B, -1)

        def clamp(xy):
            torch.clamp(xy[0], 0, W - 1, out=xy[0])
            torch.clamp(xy[1], 0, H - 1, out=xy[1])
            return xy

        # compute positive scores
        xy2p = clamp(xy2[:, None, :] + self.pos_offsets[:, :, None])
        pscores = (s_des1[None, :, :] * des2[b, :, xy2p[1], xy2p[0]]).sum(dim=-1).t()

        pscores, pos = pscores.max(dim=1, keepdim=True)
        sel_xy2 = clamp(xy2 + self.pos_offsets[:, pos.view(-1)])
        qlt = (qlt1[b, :, y1, x1] + qlt2[b, :, sel_xy2[1], sel_xy2[0]]) / 2

        # compute negative scores
        nscores = None
        if self.neg_offsets is not None:
            xy2n = clamp(xy2[:, None, :] + self.neg_offsets[:, :, None])
            nscores = (s_des1[None, :, :] * des2[b, :, xy2n[1], xy2n[0]]).sum(dim=-1).t()

        # add distractors from all images in the same mini-batch
        bd, yd, xd, _ = self.random_sampler(det2[:self.max_neg_b, ...])
        distr = des2[bd, :, yd, xd]
        dscores = torch.matmul(s_des1, distr.t())

        # remove scores that corresponds to positives (in same image)
        dis2 = (xd - xy2[0][:, None]) ** 2 + (yd - xy2[1][:, None]) ** 2
        dis2 += (bd != b[:, None]).long() * self.pos_r ** 2
        dscores[dis2 < self.pos_r ** 2] = 0

        scores = torch.cat((pscores, dscores) if self.neg_offsets is None else (pscores, nscores, dscores), dim=1)
        labels = scores.new_zeros(scores.shape, dtype=torch.bool)
        labels[:, :pscores.shape[1]] = 1
        return scores, labels, mask, qlt


class DetectionSampler(torch.nn.Module):
    """
    Sample all detection maps by dividing them to certain size cells and selecting pixels from each based on their
    detection score. Sample descriptors at these locations and calculate distance matrices between each pair.
    """

    def __init__(self, cell_d=8, border=16, random=1.0, max_b=8, blocks=True, sample_matches=False, prob_input=False):
        """
        :param cell_d: diameter of rectangular cell that is searched for max detection score
        :param border: border width, don't sample if closer than this to image borders
        :param random: if True, sample randomly instead of using the detection scores
        :param max_b:  if > 1, matches are calculated across other pairs (n-1) in the same mini-batch
        """
        super(DetectionSampler, self).__init__()
        self.random_sampler = WeightedRandomSampler(cell_d, border, random, subsample=True, act_logp=not prob_input)
        self.max_b = max_b
        self.blocks = blocks
        self.prob_input = prob_input
        self.sample_matches = sample_matches
        self.des_norm = 2
        self.px_norm = 2

    @property
    def border(self):
        return self.random_sampler.border

    @border.setter
    def border(self, border):
        self.random_sampler.border = border

    def find_matches(self, b1, x1, y1, aflow, det2):
        B, _, H, W = aflow.shape

        xy2_gt = (aflow[b1, :, y1, x1] + 0.5).long().t()
        mask = ((0 <= xy2_gt[0]) * (0 <= xy2_gt[1]) * (xy2_gt[0] < W) * (xy2_gt[1] < H))
        b2, x2, y2 = b1[mask], xy2_gt[0, mask], xy2_gt[1, mask]

        d_det = F.pixel_unshuffle(det2, self.random_sampler.cell_d)
        dist = Categorical(**{'logits' if not self.prob_input else 'probs': d_det.permute((0, 2, 3, 1))}).logits
        logp2 = F.pixel_shuffle(dist.permute((0, 3, 1, 2)), self.random_sampler.cell_d)[b2, 0, y2, x2]

        # sample matches only, normalize probabilities
        logp2 = logp2 + Bernoulli(**{'logits' if not self.prob_input else 'probs': det2[b2, 0, y2, x2]}).logits

        return b2, x2, y2, logp2

    def forward(self, output1, output2, aflow):
        des1, det1, *_ = output1
        des2, det2, *_ = output2

        B, _, H, W = aflow.shape
        D = des1.shape[1]

        # sanitize
        if self.prob_input:
            det1 = torch.nan_to_num(det1, 0.0, 1.0, 0.0)
            det2 = torch.nan_to_num(det2, 0.0, 1.0, 0.0)
        else:
            det1 = torch.nan_to_num(det1, torch.finfo(det1.dtype).min)
            det2 = torch.nan_to_num(det2, torch.finfo(det2.dtype).min)

        b1, y1, x1, logp1 = self.random_sampler(det1)

        if self.sample_matches:
            b2, y2, x2, logp2 = self.find_matches(b1, x1, y1, aflow, det2)
        else:
            b2, y2, x2, logp2 = self.random_sampler(det2)

        # sample_logp is the log p(x=sample) for current sample, it's used for calculating DISK activation cost term (!)
        #  - would it be more reasonable to use e.g. log sum(p(x_i=sample_i))? or sum(sigmoid(det1))+sum(sigmoid(det2))?
        sample_logp = logp1.sum() + logp2.sum()

        det_logp_mxs, des_dist_mxs, px_dist_mxs, masks, b1s, b2s = [], [], [], [], [], []
        max_b = min(B, self.max_b)

        # process either by matching one batch index with max_b pairs, or
        # if block is True, matching max_b pairs with max_b pairs
        blocks = self.blocks and B % max_b == 0

        for i in range(0, B - max_b + 1, max_b) if blocks else range(B):
            js = np.array(list(range(i, i + max_b)), dtype=int) % max_b
            bi = torch.zeros_like(b1, dtype=torch.bool, device=b1.device) if blocks else torch.eq(b1, i)
            bj = torch.zeros_like(b2, dtype=torch.bool, device=b2.device)
            for j in js:
                torch.logical_or(bj, torch.eq(b2, j), out=bj)
                if blocks:
                    torch.logical_or(bi, torch.eq(b1, j), out=bi)

            det_logp_mx = logp1[bi, None] + logp2[None, bj]
            det_logp_mxs.append(det_logp_mx)

            n, m = det_logp_mx.shape
            _b1, _x1, _y1 = b1[bi], x1[bi], y1[bi]
            _b2, _x2, _y2 = b2[bj], x2[bj], y2[bj]
            b1s.append(_b1)
            b2s.append(_b2)

            s_des1 = des1[_b1, :, _y1, _x1].t()     # [D, n]
            s_des2 = des2[_b2, :, _y2, _x2].t()     # [D, m]
            des_dist_mx = torch.linalg.norm(s_des1.view((D, n, 1)).expand((D, n, m))
                                            - s_des2.view((D, 1, m)).expand((D, n, m)),
                                            ord=self.des_norm, dim=0)
            des_dist_mxs.append(des_dist_mx)

            xy2 = torch.stack((_x2, _y2), dim=0).type(s_des1.dtype)
            xy2_gt = aflow[_b1, :, _y1, _x1].t()
            mask = ((0 <= xy2_gt[0]) * (0 <= xy2_gt[1]) * (xy2_gt[0] < W) * (xy2_gt[1] < H))
            masks.append(mask)

            px_dist_mx = torch.linalg.norm(xy2_gt.view((2, n, 1)).expand((2, n, m))
                                           - xy2.view((2, 1, m)).expand((2, n, m)),
                                           ord=self.px_norm, dim=0)
            px_dist_mxs.append(px_dist_mx)

        return det_logp_mxs, des_dist_mxs, px_dist_mxs, masks, b1s, b2s, sample_logp


class WeightedRandomSampler(torch.nn.Module):
    """
    Sample detection maps by dividing them to certain size cells and randomly selecting a pixel, weighted by
    the detection scores. The degree of randomness can be adjusted. Doesn't sample too close to the image borders.
    """

    def __init__(self, cell_d, border, random, subsample=False, act_logp=True):
        super(WeightedRandomSampler, self).__init__()
        self.cell_d = cell_d
        self.border = border
        self.random = random
        self.subsample = subsample  # subsample (bernoulli) from among the suggested pixels based on det
        self.act_logp = act_logp    # are det activations treated as un-normalized log probabilities or as probabilities
        self._unit_aflow = None

    def forward(self, det):
        B, _, H, W = det.shape

        if 0:
            # TODO: support all image sizes
            ml, mr, mt, mb = self.margins(W, H)
        else:
            ml, mr, mt, mb = [self.border] * 4

        n = (W - ml - mr) * (H - mt - mb) // self.cell_d ** 2
        b = torch.arange(B, device=det.device)[:, None].expand(B, n).reshape(-1)
        log_p = 0

        if self.random == float('inf'):
            # completely random, independent of activation levels
            x = torch.randint(ml, W - mr, (n,), device=det.device)
            y = torch.randint(mt, H - mb, (n,), device=det.device)
            x = x[None, :].expand(B, n).reshape(-1)
            y = y[None, :].expand(B, n).reshape(-1)
            s_det = det[b, 0, y, x]

        else:
            if self._unit_aflow is None:
                self._unit_aflow = torch.LongTensor(unit_aflow(W, H)).permute((2, 0, 1)).to(det.device)

            d_det = F.pixel_unshuffle(det[:B, :, mt:-mb, ml:-mr], self.cell_d)
            d_xy = F.pixel_unshuffle(self._unit_aflow[None, :, None, mt:-mb, ml:-mr].expand(B, -1, 1, -1, -1), self.cell_d)

            if not self.random:
                # completely deterministic, select the pixel with highest activation level within each cell
                idxs = torch.argmax(d_det, dim=1, keepdim=True)
                s_det = d_det[idxs].flatten()

                dI = idxs[:, None, :, :, :].expand(-1, 2, 1, -1, -1)
                s_xy = torch.gather(d_xy, 2, dI)[:, :, 0, :, :]
                x, y = s_xy[:, 0, :, :].reshape(-1), s_xy[:, 1, :, :].reshape(-1)

            else:
                # Cell-wise weighted random suggestions, sample accept/discard from Bernoulli distribution
                # det-activations taken to be un-normalized (log) probabilities, sample one px from each cell
                dist = Categorical(**{'logits' if self.act_logp else 'probs': d_det.permute((0, 2, 3, 1))})
                idxs = dist.sample()
                log_p = dist.log_prob(idxs).flatten()

                dI = idxs[:, None, None, :, :].expand(-1, 2, 1, -1, -1)
                s_xy = torch.gather(d_xy, 2, dI)[:, :, 0, :, :]
                x, y = s_xy[:, 0, :, :].reshape(-1), s_xy[:, 1, :, :].reshape(-1)

                # accept/reject, count accepted keeping gradient
                s_det = torch.gather(d_det, 1, idxs[:, None, :, :]).flatten()

        if self.subsample:
            dist = Bernoulli(**{'logits' if self.act_logp else 'probs': s_det})
            mask = dist.sample()
            s_det = log_p + dist.log_prob(mask)

            I = mask == 1.0
            b, x, y, s_det = b[I], x[I], y[I], s_det[I]

        return b, x, y, s_det

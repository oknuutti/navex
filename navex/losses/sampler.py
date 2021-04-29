import torch
import torch.nn.functional as F

from navex.datasets.tools import unit_aflow


class DetectionSampler(torch.nn.Module):
    """
    Sample images by dividing them to certain size cells and selecting the neighbourhood of the pixel
    with the highest detection score as positive samples. Don't sample too close to the image borders.

    Get extra negative samples from other images in the same batch.

    Distance to ground truth: 0 ... pos_r ...
    Pixel label:              + + + + - - -
    """

    def __init__(self, pos_r=2, cell_d=16, border=None, max_neg_b=None):
        super(DetectionSampler, self).__init__()

        self.pos_r = pos_r
        self.cell_d = cell_d
        self.max_neg_b = max_neg_b
        self.border = border

        offsets = [
            (i, j)
            for i in range(-self.pos_r, self.pos_r + 1)
            for j in range(-self.pos_r, self.pos_r + 1)
            if i * i + j * j <= self.pos_r ** 2
        ]
        self.offsets = torch.nn.Parameter(torch.LongTensor(offsets).view(-1, 2).t(), requires_grad=False)
        self._unit_aflow = None

    def sample(self, det, max_b=None):
        B, _, H, W = det.shape

        if max_b is not None:
            B = min(max_b, B)

        if self._unit_aflow is None:
            self._unit_aflow = torch.LongTensor(unit_aflow(W, H)).permute((2, 0, 1)).to(det.device)

        if 0:
            # TODO: support all image sizes
            l, r, t, b = self.margins(W, H)
        else:
            l, r, t, b = [self.border] * 4

        d_det = F.pixel_unshuffle(det[:B, :, t:-b, l:-r], self.cell_d)
        idxs = torch.argmax(d_det, dim=1, keepdim=True)

        d_xy = F.pixel_unshuffle(self._unit_aflow[None, :, None, t:-b, l:-r].expand(B, -1, 1, -1, -1), self.cell_d)
        dI = idxs[:, None, :, :, :].expand(-1, 2, 1, -1, -1)
        s_xy = torch.gather(d_xy, 2, dI)[:, :, 0, :, :]
        x, y = s_xy[:, 0, :, :].view(-1), s_xy[:, 1, :, :].view(-1)
        n = x.size(0)
        b = torch.arange(B, device=det.device)[:, None].expand(B, n).view(-1)

        return b, x, y, n

    def forward(self, output1, output2, aflow):
        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2
        B, _, H, W = aflow.shape

        b, y1, x1, n = self.sample(det1)
        s_des1 = des1[b, :, y1, x1]

        xy2 = (aflow[b, :, y1, x1] + 0.5).long().t()
        mask = ((0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)).view(B, n)

        def clamp(xy):
            torch.clamp(xy[0], 0, W - 1, out=xy[0])
            torch.clamp(xy[1], 0, H - 1, out=xy[1])
            return xy

        # compute positive scores
        xy2p = clamp(xy2[:, None, :] + self.offsets[:, :, None])
        pscores = (s_des1[None, :, :] * des2[b, :, xy2p[1], xy2p[0]]).sum(dim=-1).t()

        pscores, pos = pscores.max(dim=1, keepdim=True)
        sel_xy2 = clamp(xy2 + self.offsets[:, pos.view(-1)])
        qlt = (qlt1[b, :, y1, x1] + qlt2[b, :, sel_xy2[1], sel_xy2[0]]) / 2

        # add distractors from other images in same mini-batch
        bd, yd, xd, _ = self.sample(det2, max_b=self.max_neg_b)
        distr = des2[bd, :, yd, xd]
        dscores = torch.matmul(s_des1, distr.t())

        # remove scores that corresponds to positives (in same image)
        dis2 = (xd - xy2[0][:, None]) ** 2 + (yd - xy2[1][:, None]) ** 2
        dis2 += (bd != b[:, None]).long() * self.pos_r ** 2
        dscores[dis2 < self.pos_r ** 2] = 0

        scores = torch.cat((pscores, dscores), dim=1)
        labels = scores.new_zeros(scores.shape, dtype=torch.bool)
        labels[:, :pscores.shape[1]] = 1

        return scores, labels, mask, qlt

import torch
from torch.nn import Module, Unfold
from torch.functional import F


class CosSimilarityLoss(Module):
    def __init__(self, n=16, use_max=False, exclude_nans=True):
        super(CosSimilarityLoss, self).__init__()
        self.patches = Unfold(n, padding=0, stride=n // 2)
        self.use_max = use_max
        self.exclude_nans = exclude_nans    # For original R2D2 set exclude_nans = False

    def forward(self, det1, det2, aflow):
        H, W = aflow.shape[2:]

        # make both x & y in the aflow span the range [-1, 1] instead of [0, W-1] and [0, H-1]
        grid = aflow.permute(0, 2, 3, 1).clone()
        grid[:, :, :, 0] *= 2 / (W - 1)
        grid[:, :, :, 1] *= 2 / (H - 1)
        grid -= 1
        grid[grid.isnan()] = 1e6    # still, on 2022-10-26, torch grid_sample does not handle nans correctly
                                    # (https://github.com/pytorch/pytorch/issues/24823)

        warped_det2 = F.grid_sample(det2, grid, mode='bilinear', padding_mode='border', align_corners=False)

        if self.exclude_nans and not self.use_max:
            warped_det2[aflow[:, 0:1, :, :].isnan()] = float('nan')

        # patches are already normalized to unit length (possibly taking into account nans)
        patches1 = self.patches(det1).transpose(1, 2)
        patches2 = self.patches(warped_det2).transpose(1, 2)

        if self.exclude_nans and not self.use_max:
            nans = patches2.isnan()
            patches1[nans] = 0
            patches2[nans] = 0

        if self.use_max:
            cosim = (patches1 * patches2).amax(dim=2)
        else:
            patches1 = F.normalize(patches1, p=2, dim=2)
            patches2 = F.normalize(patches2, p=2, dim=2)
            cosim = (patches1 * patches2).sum(dim=2)

        return 1 - torch.mean(cosim)

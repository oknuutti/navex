import torch
from torch.nn import Module, Unfold
from torch.functional import F


class CosSimilarityLoss(Module):
    def __init__(self, n=16, use_max=False, ignore_nan=True):
        super(CosSimilarityLoss, self).__init__()
        self.patches = Unfold(n, padding=0, stride=n // 2)
        self.use_max = use_max
        self.ignore_nan = ignore_nan

    def extract_patches(self, det):
        patches = self.patches(det).transpose(1, 2)
        if self.use_max:
            pass
        elif not self.ignore_nan:
            patches = F.normalize(patches, p=2, dim=2)
        else:
            norm = torch.sqrt(patches.pow(2).nansum(dim=2, keepdim=True))
            patches = patches / norm
        return patches

    def forward(self, det1, det2, aflow):
        H, W = aflow.shape[2:]

        # make both x & y in the aflow span the range [-1, 1] instead of [0, W-1] and [0, H-1]
        grid = aflow.permute(0, 2, 3, 1).clone()
        grid[:, :, :, 0] *= 2 / (W - 1)
        grid[:, :, :, 1] *= 2 / (H - 1)
        grid -= 1
        grid[grid.isnan()] = 1e6    # still, on 2022-10-26, torch grid_sample does not handle nans correctly
                                    # (https://github.com/pytorch/pytorch/issues/24823)

        warped_det2 = F.grid_sample(det2, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        if self.ignore_nan and not self.use_max:
            warped_det2[aflow[:, 0:1, :, :].isnan()] = float('nan')

        # patches are already normalized to unit length
        patches1 = self.extract_patches(det1)
        patches2 = self.extract_patches(warped_det2)

        if self.use_max:
            cosim = (patches1 * patches2).amax(dim=2)
            loss = 1 - torch.mean(cosim)
        elif not self.ignore_nan:
            cosim = (patches1 * patches2).sum(dim=2)
            loss = 1 - torch.mean(cosim)
        else:
            cosim = (patches1 * patches2).nansum(dim=2)
            loss = 1 - torch.nansum(cosim) / max(1, cosim.isnan().logical_not().sum())

        return loss

import torch
from torch.nn import Module, Unfold
from torch.functional import F


class CosSimilarityLoss(Module):
    def __init__(self, n=16, use_max=False, implicit_nan_exclusion=False):
        super(CosSimilarityLoss, self).__init__()
        self.patches = Unfold(n, padding=0, stride=n // 2)
        self.use_max = use_max
        self.implicit_nan_exclusion = implicit_nan_exclusion

    def extract_patches(self, det):
        patches = self.patches(det).transpose(1, 2)
        if self.use_max:
            pass
        elif self.implicit_nan_exclusion:
            patches = F.normalize(patches, p=2, dim=2)
        else:
            patches[patches.isnan()] = 0
            norm2 = patches.pow(2).sum(dim=2, keepdim=True)
            norm2[norm2 == 0] = 1
            patches = patches / torch.sqrt(norm2)
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

        if not self.implicit_nan_exclusion and not self.use_max:
            warped_det2[aflow[:, 0:1, :, :].isnan()] = float('nan')

        # patches are already normalized to unit length (possibly taking into account nans)
        patches1 = self.extract_patches(det1)
        patches2 = self.extract_patches(warped_det2)

        if self.use_max:
            cosim = (patches1 * patches2).amax(dim=2)
            loss = 1 - torch.mean(cosim)
        elif self.implicit_nan_exclusion:
            cosim = (patches1 * patches2).sum(dim=2)
            loss = 1 - torch.mean(cosim)
        else:
            mask = patches2.isnan()
            patches2[mask] = 0
            cosim = (patches1 * patches2).sum(dim=2)
            valids = mask.logical_not().sum(dim=2) > 0
            loss = 1 - cosim.sum() / max(1, valids.sum())  # ignore patches with all nans

        return loss

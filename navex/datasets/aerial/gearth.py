import math
import os

from ..base import SynthesizedPairDataset, DatabaseImagePairDataset, AugmentedPairDatasetMixin, RandomSeed
from ..transforms import RandomHomography2


class GoogleEarthPairDataset(DatabaseImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='gearth', noise_max=0.0, rnd_gain=1.0, image_size=512,
                 margin=16, max_tr=0, max_rot=math.radians(8), max_shear=0.2, max_proj=0.4,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, eval=eval, rgb=False,
                                           resize_max_size=None, resize_max_sc=None, blind_crop=False)
        DatabaseImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms)

        self.pre_transf = RandomHomography2(max_tr=max_tr, max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                                            min_size=image_size//2, crop_valid=True)

    def preprocess(self, idx, imgs, aflow):
        eval = getattr(self, 'eval', False)
        if eval:
            with RandomSeed(idx):
                imgs, aflow = self.pre_transf(imgs, aflow)
        else:
            imgs, aflow = self.pre_transf(imgs, aflow)
        return imgs, aflow

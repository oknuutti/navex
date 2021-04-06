import os
import math

from .base import AsteroidSynthesizedPairDataset, AsteroidImagePairDataset
from ..base import AugmentedPairDatasetMixin
from ..tools import find_files_recurse, ImageDB


class CG67pNavcamSynthPairDataset(AsteroidSynthesizedPairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='cg67p/navcam', max_tr=0, max_rot=math.radians(15), max_shear=0.2, max_proj=0.8,
                 noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4), margin=16,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           margin=margin, max_sc=max_sc, fill_value=0, eval=eval, rgb=False, blind_crop=True)

        AsteroidSynthesizedPairDataset.__init__(self, os.path.join(root, folder), max_tr=max_tr,
                                                max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                                                min_size=image_size//2, transforms=self.transforms)

    def _load_samples(self):
        return find_files_recurse(self.root, ext='.png')


class CG67pOsinacPairDataset(AsteroidImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='cg67p/osinac', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, eval=eval, rgb=False,
                                           resize_max_sc=None, blind_crop=True)
        AsteroidImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms,
                                          trg_north_ra=math.radians(69.3), trg_north_dec=math.radians(64.1),
                                          model_north=[0, 1, 0])
        # axis ra & dec from http://www.esa.int/ESA_Multimedia/Images/2015/01/Comet_vital_statistics

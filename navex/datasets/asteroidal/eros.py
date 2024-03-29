import math
import os

from .base import AsteroidImagePairDataset, not_aflow_file
from ..base import AugmentedPairDatasetMixin, BasicDataset
from ..preproc.eros_msi import CAM


class ErosPairDataset(AsteroidImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='eros', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 aflow_rot_norm=False, margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, resize_max_sc=1.0,
                                           eval=eval, rgb=False, blind_crop=False)
        AsteroidImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms,
                                          aflow_rot_norm=aflow_rot_norm, extra_crop=[0, 0, 10, 6],
                                          trg_north_ra=math.radians(11.38), trg_north_dec=math.radians(17.18),
                                          model_north=[1, 0, 0], cam=CAM)
        # axis ra & dec from https://science.sciencemag.org/content/289/5487/2097/tab-figures-data


class ErosDataset(BasicDataset):
    def __init__(self, root='data', folder='eros', **kwargs):
        super(ErosDataset, self).__init__(root, folder, ext='.png', folder_depth=2, test=not_aflow_file, **kwargs)

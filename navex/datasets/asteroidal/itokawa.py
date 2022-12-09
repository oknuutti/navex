import math
import os

from .base import AsteroidImagePairDataset, not_aflow_file
from ..base import AugmentedPairDatasetMixin, BasicDataset
from ..preproc.itokawa_amica import CAM


class ItokawaPairDataset(AsteroidImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='itokawa', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 aflow_rot_norm=False, margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           margin=margin, max_sc=1.0, fill_value=0, eval=eval, rgb=False,
                                           resize_max_sc=1.0, blind_crop=False)
        AsteroidImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms,
                                          aflow_rot_norm=aflow_rot_norm,
                                          trg_north_ra=math.radians(90.53), trg_north_dec=math.radians(-66.30),
                                          model_north=[0, 0, 1], cam=CAM)


class ItokawaDataset(BasicDataset):
    def __init__(self, root='data', folder='itokawa', **kwargs):
        super(ItokawaDataset, self).__init__(root, folder, ext='.png', folder_depth=2, test=not_aflow_file, **kwargs)

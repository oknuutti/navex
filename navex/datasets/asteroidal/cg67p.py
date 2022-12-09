import os
import math

from .base import AsteroidSynthesizedPairDataset, AsteroidImagePairDataset, not_aflow_file
from ..base import AugmentedPairDatasetMixin, BasicDataset
from ..preproc.cg67p_rosetta import INSTR
from ..tools import find_files_recurse, ImageDB


class CG67pNavcamSynthPairDataset(AsteroidSynthesizedPairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='cg67p/navcam', max_tr=0, max_rot=math.radians(15), max_shear=0.2, max_proj=0.8,
                 noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4), margin=16,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           margin=margin, max_sc=max_sc, fill_value=0, eval=eval, rgb=False, blind_crop=False)

        AsteroidSynthesizedPairDataset.__init__(self, os.path.join(root, folder), max_tr=max_tr,
                                                max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                                                min_size=image_size//2, transforms=self.transforms,
                                                warp_crop=False)

    def _load_samples(self):
        return find_files_recurse(self.root, ext='.png')


class CG67pOsinacPairDataset(AsteroidImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='cg67p/osinac', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 aflow_rot_norm=False, margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, eval=eval, rgb=False,
                                           resize_max_sc=1.0, blind_crop=False)
        AsteroidImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms,
                                          aflow_rot_norm=aflow_rot_norm, extra_crop=[3, 10, 0, 0],
                                          trg_north_ra=math.radians(69.3), trg_north_dec=math.radians(64.1),
                                          model_north=[0, 1, 0], cam=INSTR['osinac']['cam'])
        # axis ra & dec from http://www.esa.int/ESA_Multimedia/Images/2015/01/Comet_vital_statistics


class CG67pNavcamDataset(BasicDataset):
    def __init__(self, root='data', folder='cg67p/navcam', **kwargs):
        super(CG67pNavcamDataset, self).__init__(root, folder, ext='.png', folder_depth=2, test=not_aflow_file, **kwargs)


class CG67pOsinacDataset(BasicDataset):
    def __init__(self, root='data', folder='cg67p/osinac', **kwargs):
        super(CG67pOsinacDataset, self).__init__(root, folder, ext='.png', folder_depth=2, test=not_aflow_file, **kwargs)

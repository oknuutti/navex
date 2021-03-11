import os
import math

from ..base import SynthesizedPairDataset, AugmentedPairDatasetMixin
from ..tools import find_files_recurse


class BennuSynthPairDataset(SynthesizedPairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='bennu/tagcams', max_tr=0, max_rot=math.radians(15), max_shear=0.2, max_proj=0.8,
                 noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4),
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=max_sc, eval=eval, rgb=False, blind_crop=True)

        SynthesizedPairDataset.__init__(self, os.path.join(root, folder), max_tr=max_tr,
                                        max_rot=max_rot, max_shear=max_shear, max_proj=max_proj, min_size=image_size//2,
                                        transforms=self.transforms)

    def _load_samples(self):
        return find_files_recurse(self.root)

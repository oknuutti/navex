import itertools
import math
import os

from ..base import SynthesizedPairDataset, DatabaseImagePairDataset, AugmentedPairDatasetMixin
from ..tools import find_files_recurse


class BatvikSynthPairDataset(SynthesizedPairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='batvik', subset=None, max_tr=0, max_rot=math.radians(15), max_shear=0.2,
                 max_proj=0.8, noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4), margin=16,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        self.folder = folder
        self.subset = subset

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=max_sc, margin=margin, eval=eval, rgb=rgb, blind_crop=True)

        SynthesizedPairDataset.__init__(self, os.path.join(root, self.folder), max_tr=max_tr,
                                        max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                                        min_size=image_size // 2)

    def _load_samples(self):
        paths = [self.root]
        if self.subset is not None:
            paths = [os.path.join(self.root, s) for s in self.subset]
        return list(itertools.chain(*[find_files_recurse(path, ext='.jpg') for path in paths]))


class BatvikPairDataset(DatabaseImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='batvik', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, eval=eval, rgb=False,
                                           resize_max_sc=1.0, blind_crop=False)
        DatabaseImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms)

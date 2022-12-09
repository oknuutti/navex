import os

from .base import AsteroidImagePairDataset, not_aflow_file
from ..base import AugmentedPairDatasetMixin, BasicDataset
from ..preproc.synth import CAM


class SynthBennuPairDataset(AsteroidImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='synth', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, eval=eval, rgb=False,
                                           blind_crop=False)
        AsteroidImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms, cam=CAM)


class SynthBennuDataset(BasicDataset):
    def __init__(self, root='data', folder='synth', **kwargs):
        super(SynthBennuDataset, self).__init__(root, folder, ext='.png', folder_depth=2, test=not_aflow_file, **kwargs)

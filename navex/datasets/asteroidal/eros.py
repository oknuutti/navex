import os

from ..base import ImagePairDataset, AugmentedPairDatasetMixin


class ErosPairDataset(ImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='eros', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, eval=eval, rgb=False, blind_crop=True)
        ImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms)

import os

from ..base import SynthesizedPairDataset, DatabaseImagePairDataset, AugmentedPairDatasetMixin


class GoogleEarthPairDataset(DatabaseImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='gearth', noise_max=0.0, rnd_gain=1.0, image_size=512,
                 margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, eval=eval, rgb=False,
                                           resize_max_size=None, resize_max_sc=None, blind_crop=False)
        DatabaseImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms)

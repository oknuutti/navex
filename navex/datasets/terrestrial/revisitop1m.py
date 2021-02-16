import math
import os

from torchvision.datasets.folder import default_loader

from navex.datasets.base import SynthesizedPairDataset, AugmentedPairDatasetMixin, BasicDataset
from navex.datasets.tools import find_files_recurse


class WebImageSynthPairDataset(SynthesizedPairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='revisitop1m', max_tr=0, max_rot=math.radians(15), max_shear=0.2, max_proj=0.8,
                 noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4),
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        self.npy = npy
        self.min_size = int(image_size*0.75)
        self.max_aspect_ratio = 2.1

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=max_sc, eval=eval, rgb=rgb, blind_crop=True)

        SynthesizedPairDataset.__init__(self, os.path.join(root, folder), max_tr=max_tr, max_rot=max_rot,
                                        max_shear=max_shear, max_proj=max_proj, min_size=image_size//2,
                                        transforms=self.transforms)

    def _load_samples(self, test=False):
        s = self

        def test_fn(imgpath):
            img = default_loader(imgpath)
            return (img.size[0] >= s.min_size and img.size[1] >= s.min_size
                    and max(img.size) / min(img.size) <= s.max_aspect_ratio)

        samples = find_files_recurse(self.root, ext='.jpg', npy=self.npy, test=test_fn if test else None)
        return samples


class WebImageDataset(BasicDataset):
    def __init__(self, root='data', folder='revisitop1m', **kwargs):
        kwargs['folder_depth'] = 100
        super(WebImageDataset, self).__init__(root, folder, **kwargs)

import math
import os

from navex.datasets.base import SynthesizedPairDataset, AugmentedDatasetMixin


class WebImageSynthPairDataset(SynthesizedPairDataset, AugmentedDatasetMixin):
    def __init__(self, root='data', folder='revisitop1m', max_tr=0, max_rot=math.radians(15), max_shear=0.2, max_proj=0.8,
                 noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4),
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        self.npy = npy
        self.min_size = int(image_size*0.75)
        self.max_aspect_ratio = 2.5

        AugmentedDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                       max_sc=max_sc, eval=eval, rgb=rgb, blind_crop=True)

        SynthesizedPairDataset.__init__(self, os.path.join(root, folder), max_tr=max_tr, max_rot=max_rot,
                                        max_shear=max_shear, max_proj=max_proj, min_size=image_size//2,
                                        transforms=self.transforms)

    def _recurse(self, path, samples, test):
        for fname in os.listdir(path):
            fullpath = os.path.join(path, fname)
            if fname[-4:] == ('.npy' if self.npy else '.jpg'):
                ok = not test
                if test:
                    try:
                        img = self.image_loader(fullpath)
                        ok = (img.size[0] >= self.min_size and img.size[1] >= self.min_size
                              and max(img.size)/min(img.size) <= self.max_aspect_ratio)
                    except Exception as e:
                        print('%s' % e)
                if ok:
                    samples.append(fullpath)
                else:
                    print('rejected: %s' % fullpath)
            elif os.path.isdir(fullpath):
                self._recurse(fullpath, samples, test)

    def _load_samples(self, test=True):
        samples = []
        self._recurse(self.root, samples, test)
        samples = sorted(samples)
        return samples

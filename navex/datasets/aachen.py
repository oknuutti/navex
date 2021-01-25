import os
import math

import numpy as np

from r2d2.datasets.aachen import AachenPairs_OpticalFlow

from .base import ImagePairDataset, DataLoadingException, AugmentedDatasetMixin, SynthesizedPairDataset, unit_aflow


class AachenFlowDataset(AachenPairs_OpticalFlow, ImagePairDataset, AugmentedDatasetMixin):
    def __init__(self, root='data', folder='aachen', noise_max=0.25, rnd_gain=(0.5, 3), image_size=512, max_sc=2 ** (1 / 4),
                 eval=False, rgb=False, npy=False):

        root = os.path.join(root, folder)
        AachenPairs_OpticalFlow.__init__(self, root, rgb=rgb, npy=npy)
        AugmentedDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                       max_sc=max_sc, eval=eval, rgb=rgb)
        ImagePairDataset.__init__(self, root, None, transforms=self.transforms)

    def _load_samples(self):
        s = list(range(self.npairs))
        return zip(s, s)

    def __getitem__(self, idx):
        img1, img2, meta = self.get_pair(idx, output=('aflow', 'mask'))
        aflow = meta['aflow'].astype(np.float32)
        aflow[np.logical_not(meta['mask'])] = np.nan

        try:
            (img1, img2), aflow = self.transforms((img1, img2), aflow)
        except Exception as e:
            raise DataLoadingException("Problem with dataset %s, index %s: %s" %
                                       (self.__class__, idx, self.samples[idx],)) from e

        return (img1, img2), aflow


class AachenStyleTransferDataset(ImagePairDataset, AugmentedDatasetMixin):
    def __init__(self, root='data', folder='aachen', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'

        AugmentedDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                       max_sc=1.0, eval=eval, rgb=rgb, blind_crop=True)

        ImagePairDataset.__init__(self, os.path.join(root, folder), self.identity_aflow, transforms=self.transforms)
        self.npy = npy

    @staticmethod
    def identity_aflow(path, img1_size, img2_size):
        sc = 0.5 * (img2_size[0]/img1_size[0] + img2_size[1]/img1_size[1])
        return unit_aflow(*img1_size) * sc

    def _load_samples(self):
        path_db = os.path.join(self.root, 'images_upright', 'db')
        path_st = os.path.join(self.root, 'style_transfer')

        samples = []
        for file_st in os.listdir(path_st):
            if file_st[-4:] == '.jpg':
                file_db = file_st.split('.jpg.st_')[0]
                samples.append(((os.path.join(path_db, file_db + '.jpg'), os.path.join(path_st, file_st)), None))

        samples = sorted(samples, key=lambda x: x[0][1])
        return samples


class AachenSynthPairDataset(SynthesizedPairDataset, AugmentedDatasetMixin):
    def __init__(self, root='data', folder='aachen', max_tr=0, max_rot=math.radians(15), max_shear=0.2, max_proj=0.8,
                 noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4),
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        self.npy = npy

        AugmentedDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                       max_sc=max_sc, eval=eval, rgb=rgb, blind_crop=True)

        SynthesizedPairDataset.__init__(self, os.path.join(root, folder), max_tr=max_tr, max_rot=max_rot,
                                        max_shear=max_shear, max_proj=max_proj, transforms=self.transforms)

    def _load_samples(self):
        path_db = os.path.join(self.root, 'images_upright', 'db')

        samples = []
        for file_db in os.listdir(path_db):
            if file_db[-4:] == ('.npy' if self.npy else '.jpg'):
                samples.append(os.path.join(path_db, file_db))

        samples = sorted(samples)
        return samples

import os
import math

import numpy as np

from r2d2.datasets.aachen import AachenPairs_OpticalFlow

from navex.datasets.base import ImagePairDataset, DataLoadingException, AugmentedPairDatasetMixin, \
    SynthesizedPairDataset, \
    BasicDataset, RandomSeed
from navex.datasets.tools import find_files, unit_aflow
from navex.datasets.transforms import RandomTiltWrapper2, RandomHomography2


class AachenFlowPairDataset(AachenPairs_OpticalFlow, ImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='aachen', noise_max=0.1, rnd_gain=(0.5, 2), image_size=512,
                 max_sc=2 ** (1 / 4), margin=16, eval=False, rgb=False, npy=False):

        root = os.path.join(root, folder)
        AachenPairs_OpticalFlow.__init__(self, root, rgb=rgb, npy=npy)
        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=max_sc, margin=margin, eval=eval, rgb=rgb, blind_crop=False)
        ImagePairDataset.__init__(self, root, None, transforms=self.transforms)

    def _load_samples(self):
        return [tuple(map(self.get_filename, pair)) for pair in self.image_pairs]

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


class AachenStyleTransferPairDataset(ImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='aachen', noise_max=0.1, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/8),
                 max_tr=0, max_rot=math.radians(8), max_shear=0.2, max_proj=0.4, margin=16,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=max_sc, margin=margin, eval=eval, rgb=rgb, blind_crop=True)

        ImagePairDataset.__init__(self, os.path.join(root, folder), self.identity_aflow, transforms=self.transforms)

        # self.pre_transf = RandomTiltWrapper2(magnitude=0.5)
        self.pre_transf = RandomHomography2(max_tr=max_tr, max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                                            min_size=image_size//2, crop_valid=True)
        self.npy = npy

    def preprocess(self, idx, imgs, aflow):
        eval = getattr(self, 'eval', False)
        if eval:
            with RandomSeed(idx):
                imgs, aflow = self.pre_transf(imgs, aflow)
        else:
            imgs, aflow = self.pre_transf(imgs, aflow)
        return imgs, aflow

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


class AachenSynthPairDataset(SynthesizedPairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='aachen', max_tr=0, max_rot=math.radians(8), max_shear=0.2, max_proj=0.4,
                 noise_max=0.1, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4), margin=16,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        self.npy = npy

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=max_sc, margin=margin, eval=eval, rgb=rgb, blind_crop=True)

        SynthesizedPairDataset.__init__(self, os.path.join(root, folder, 'images_upright', 'db'), max_tr=max_tr,
                                        max_rot=max_rot, max_shear=max_shear, max_proj=max_proj, min_size=image_size//2,
                                        transforms=self.transforms, warp_crop=True)

    def _load_samples(self):
        return find_files(self.root, self.npy)


class AachenDataset(BasicDataset):
    def __init__(self, root='data', folder='aachen', **kwargs):
        folder = os.path.join(folder, 'images_upright', 'db')
        super(AachenDataset, self).__init__(root, folder, **kwargs)


class AachenSyntheticNightDataset(BasicDataset):
    def __init__(self, root='data', folder='aachen', **kwargs):
        folder = os.path.join(folder, 'style_transfer')
        super(AachenSyntheticNightDataset, self).__init__(root, folder, **kwargs)

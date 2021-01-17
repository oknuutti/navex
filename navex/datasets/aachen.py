from copy import copy
import os

import numpy as np
import torchvision.transforms as tr

from r2d2.datasets.aachen import AachenPairs_OpticalFlow
from torch.utils.data.dataset import random_split

from .base import RandomDarkNoise, RandomExposure, ImagePairDataset, PhotometricTransform, ComposedTransforms, \
    GeneralTransform, PairedCenterCrop, PairedRandomCrop, DataLoadingException, PairedIdentityTransform


class AachenFlowDataset(AachenPairs_OpticalFlow, ImagePairDataset):
    TR_NORM_RGB = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __init__(self, root, noise_max=0.25, rnd_gain=(0.5, 3), image_size=512, max_sc=2**(1/4), eval=False, rgb=False, npy=False):
        AachenPairs_OpticalFlow.__init__(self, root, rgb=rgb, npy=npy)
        self.rgb = rgb
        self.image_size = image_size
        self.max_sc = max_sc
        self.noise_max = noise_max
        self.rnd_gain = rnd_gain if isinstance(rnd_gain, (tuple, list)) else (1 / rnd_gain, rnd_gain)

        if eval:
            transforms = self._eval_transf()
        else:
            transforms = ComposedTransforms([
                PhotometricTransform(tr.Grayscale(num_output_channels=1)) if not self.rgb else PairedIdentityTransform(),
                PairedRandomCrop(self.image_size, max_sc_diff=self.max_sc),
                GeneralTransform(tr.ToTensor()),
                PhotometricTransform(RandomDarkNoise(0, self.noise_max, 0.3, 3)),  # apply extra dark noise at a random level (dropout might be enough though)
                PhotometricTransform(RandomExposure(*self.rnd_gain)),  # apply a random gain on the image
                PhotometricTransform(tr.Normalize(mean=[0.449], std=[0.226]) if not self.rgb else self.TR_NORM_RGB),
            ])

        ImagePairDataset.__init__(self, root, None, transforms=transforms)

    def _eval_transf(self):
        return ComposedTransforms([
            PhotometricTransform(tr.Grayscale(num_output_channels=1)) if not self.rgb else PairedIdentityTransform(),
            PairedCenterCrop(self.image_size, max_sc_diff=self.max_sc),
            GeneralTransform(tr.ToTensor()),
            PhotometricTransform(tr.Normalize(mean=[0.449], std=[0.226]) if not self.rgb else self.TR_NORM_RGB),
        ])

    def _load_samples(self):
        s = list(range(self.npairs))
        return zip(s, s)

    def __getitem__(self, idx):
        img1, img2, meta = self.get_pair(idx, output=('aflow', 'mask'))
        aflow = meta['aflow'].astype(np.float32)
        aflow[np.logical_not(meta['mask'])] = np.nan

        try:
            if self.transforms is not None:
                (img1, img2), aflow = self.transforms((img1, img2), aflow)
        except DataLoadingException as e:
            raise DataLoadingException("Problem with idx %s:\n%s" % (idx, self.image_pairs[idx],)) from e

        return (img1, img2), aflow

    def split(self, *ratios, eval=tuple(), rgb=False):
        assert np.isclose(np.sum(ratios), 1.0), 'the ratios do not sum to one'

        eval_ds = self
        if eval:
            eval_ds = copy(self)    # shallow copy should be enough
            eval_ds.transforms = self._eval_transf()

        total = len(self)
        lengths = []
        for i, r in enumerate(ratios):
            n = round(total * r)
            lengths.append((total-np.sum(lengths)) if len(ratios)-1 == i else n)

        datasets = random_split(self, lengths)
        for i in eval:
            datasets[i].dataset = eval_ds

        return datasets

from copy import copy
import os

import numpy as np
import torchvision.transforms as tr

from r2d2.datasets.aachen import AachenPairs_OpticalFlow
from torch.utils.data.dataset import random_split

from .base import RandomDarkNoise, RandomExposure, ImagePairDataset, PhotometricTransform, ComposedTransforms, \
    GeneralTransform, PairedCenterCrop, PairedRandomCrop, DataLoadingException, IdentityTransform


class AachenFlowDataset(AachenPairs_OpticalFlow, ImagePairDataset):
    TRFM_EVAL = ComposedTransforms([
        PhotometricTransform(tr.Grayscale(num_output_channels=1)),
        PairedCenterCrop(512, max_sc_diff=2**(1/4)),
        GeneralTransform(tr.ToTensor()),
        PhotometricTransform(tr.Normalize(mean=[0.449], std=[0.226])),
    ])

    def __init__(self, root, noise_max=0.25, rnd_gain=(0.5, 3), eval=False, rgb=False, npy=False):
        AachenPairs_OpticalFlow.__init__(self, root, rgb=rgb, npy=npy)

        if eval:
            transforms = self.TRFM_EVAL
        else:
            if not isinstance(rnd_gain, (tuple, list)):
                rnd_gain = (1/rnd_gain, rnd_gain)

            transforms = ComposedTransforms([
                PhotometricTransform(tr.Grayscale(num_output_channels=1)),
                PairedRandomCrop(512, max_sc_diff=2**(1/4)),
                GeneralTransform(tr.ToTensor()),
                PhotometricTransform(RandomDarkNoise(0, noise_max, 0.3, 3)),  # apply extra dark noise at a random level (dropout might be enough though)
                PhotometricTransform(RandomExposure(*rnd_gain)),  # apply a random gain on the image
                PhotometricTransform(tr.Normalize(mean=[0.449], std=[0.226])),
            ])

        if rgb:
            transforms.transforms[2] = IdentityTransform()

        ImagePairDataset.__init__(self, root, None, transforms=transforms)

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
            eval_ds.transforms = self.TRFM_EVAL
            if rgb:
                eval_ds.transforms.transforms[2] = IdentityTransform()

        total = len(self)
        lengths = []
        for i, r in enumerate(ratios):
            n = round(total * r)
            lengths.append((total-np.sum(lengths)) if len(ratios)-1 == i else n)

        datasets = random_split(self, lengths)
        for i in eval:
            datasets[i].dataset = eval_ds

        return datasets

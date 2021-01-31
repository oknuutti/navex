import bisect
import os
import random

import numpy as np
import torch

from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms as tr

from .transforms import RandomDarkNoise, RandomExposure, PhotometricTransform, ComposedTransforms, \
    GeneralTransform, PairedCenterCrop, PairedRandomCrop, PairedIdentityTransform, RandomHomography, IdentityTransform, \
    RandomTiltWrapper, RandomScale, ScaleToRange

from .. import RND_SEED


def worker_init_fn(id):
    random.seed(RND_SEED)
    np.random.seed(RND_SEED)
    torch.random.manual_seed(RND_SEED)


def unit_aflow(W, H):
    return np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)), axis=2)


class DataLoadingException(Exception):
    pass


class ImagePairDataset(VisionDataset):
    def __init__(self, root, aflow_loader, image_loader=default_loader, transforms=None):
        super(ImagePairDataset, self).__init__(root, transforms=transforms)
        self.aflow_loader = aflow_loader
        self.image_loader = image_loader
        self.samples = self._load_samples()

    def __getitem__(self, idx):
        (img1_pth, img2_pth), aflow_pth = self.samples[idx]

        try:
            img1 = self.image_loader(img1_pth)
            img2 = self.image_loader(img2_pth)
            aflow = self.aflow_loader(aflow_pth, img1.size, img2.size)

            if self.transforms is not None:
                (img1, img2), aflow = self.transforms((img1, img2), aflow)

        except Exception as e:
            raise DataLoadingException("Problem with dataset %s, index %s: %s" %
                                       (self.__class__, idx, self.samples[idx],)) from e

        return (img1, img2), aflow

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        raise NotImplemented()


class IndexFileLoader:
    """
    Args:
        img1_col (int): index file image 1 column
        img2_col (int): index file image 2 column
        aflow_col (int): index file absolute flow column
        img_root (string): base path to prepend to image file paths
        aflow_root (string): base path to prepend to aflow file path
    """
    def __init__(self, index_file_sep=',', img1_col=0, img2_col=1, aflow_col=2, img_root=None, aflow_root=None):
        self.index_file_sep = index_file_sep
        self.img1_col = img1_col
        self.img2_col = img2_col
        self.aflow_col = aflow_col
        self.img_root = img_root
        self.aflow_root = aflow_root

    def __call__(self, file):
        samples = []
        with open(file, 'r') as fh:
            for line in fh:
                c = line.split(self.index_file_sep)
                img1_path = c[self.img1_col].strip()
                img2_path = c[self.img2_col].strip()
                aflow_path = c[self.aflow_col].strip()

                if self.img_root is not None:
                    img1_path = os.path.join(self.img_root, img1_path)
                    img2_path = os.path.join(self.img_root, img2_path)
                if self.aflow_root is not None:
                    aflow_path = os.path.join(self.aflow_root, aflow_path)

                samples.append(((img1_path, img2_path), aflow_path))

        return samples


class IndexedImagePairDataset(ImagePairDataset):
    """An image pair data loader where the index file has has three columns: ::
    img1 path, img2 path, aflow path

    Args:
        index_file (string): path of the index file
        image_loader (callable): A function to load an image given its path.
        aflow_loader (callable): A function to load an absolute flow file given its path.
        transforms (callable, optional): A function/transform that takes in
            a pair of images and corresponding absolute flow and returns a transformed version of them
            E.g, ``RandomExposure`` which doesn't have a geometric distortion.

     Attributes:
        samples (list): List of ((img1 path, img2 path), aflow path) tuples
    """
    def __init__(self, index_file, aflow_loader, image_loader=default_loader,
                 index_file_loader=IndexFileLoader(), transforms=None):
        self.index_file = index_file
        self.index_file_loader = index_file_loader
        super(IndexedImagePairDataset, self).__init__(None, aflow_loader, image_loader, transforms=transforms)

    def _load_samples(self):
        return self.index_file_loader(self.index_file)


class RandomSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        random.setstate(self.state)
        return True


class SynthesizedPairDataset(VisionDataset):
    def __init__(self, root, max_tr, max_rot, max_shear, max_proj, min_size, transforms=None, image_loader=default_loader):
        super(SynthesizedPairDataset, self).__init__(root, transforms=transforms)

        # fv = AugmentedDatasetMixin.TR_NORM_RGB.mean if self.rgb else AugmentedDatasetMixin.TR_NORM_MONO.mean
        fv = np.nan
        self.warping_transforms = tr.Compose([
            IdentityTransform() if self.rgb else tr.Grayscale(num_output_channels=1),
            RandomHomography(max_tr=max_tr, max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                             min_size=min_size, fill_value=fv),
#            RandomTiltWrapper(magnitude=0.5)
        ])

        self.image_loader = image_loader
        self.samples = self._load_samples()

    def __getitem__(self, idx):
        img_pth = self.samples[idx]

        try:
            img1 = self.image_loader(img_pth)

            eval = getattr(self, 'eval', False)
            if eval:
                with RandomSeed(idx):
                    img2, aflow = self.warping_transforms(img1)
            else:
                img2, aflow = self.warping_transforms(img1)

            if self.transforms is not None:
                (img1, img2), aflow = self.transforms((img1, img2), aflow)

        except Exception as e:
            raise DataLoadingException("Problem with dataset %s, index %s: %s" %
                                       (self.__class__, idx, self.samples[idx],)) from e

        return (img1, img2), aflow

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        raise NotImplemented()


class AugmentedDatasetMixin:
    TR_NORM_RGB = tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    TR_NORM_MONO = tr.Normalize(mean=[0.449], std=[0.226])

    def __init__(self, noise_max=0.25, rnd_gain=(0.5, 3), image_size=512, max_sc=2**(1/4), blind_crop=False,
                 eval=False, rgb=False):

        self.noise_max = noise_max
        self.rnd_gain = rnd_gain if isinstance(rnd_gain, (tuple, list)) else (1 / rnd_gain, rnd_gain)
        self.image_size = image_size
        self.max_sc = max_sc
        self.blind_crop = blind_crop
        self.eval = eval
        self.rgb = rgb

        fill_value = AugmentedDatasetMixin.TR_NORM_RGB.mean if self.rgb else AugmentedDatasetMixin.TR_NORM_MONO.mean
        self._train_transf = ComposedTransforms([
            PhotometricTransform(tr.Grayscale(num_output_channels=1)) if not self.rgb else PairedIdentityTransform(),
            RandomScale(min_size=max(self.image_size, 256), max_size=1280),
            PairedRandomCrop(self.image_size, max_sc_diff=self.max_sc, blind_crop=self.blind_crop, fill_value=fill_value),
            GeneralTransform(tr.ToTensor()),
            PhotometricTransform(RandomDarkNoise(0, self.noise_max, 0.3, 3)),  # apply extra dark noise at a random level (dropout might be enough though)
            PhotometricTransform(RandomExposure(*self.rnd_gain)),  # apply a random gain on the image
            PhotometricTransform(self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB),
        ])
        self._eval_transf = ComposedTransforms([
            PhotometricTransform(tr.Grayscale(num_output_channels=1)) if not self.rgb else PairedIdentityTransform(),
            ScaleToRange(min_size=max(self.image_size, 256), max_size=1280),
            PairedCenterCrop(self.image_size, max_sc_diff=self.max_sc, blind_crop=self.blind_crop, fill_value=fill_value),
            GeneralTransform(tr.ToTensor()),
            PhotometricTransform(self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB),
        ])
        self.transforms = self._eval_transf if self.eval else self._train_transf

    def set_eval(self, eval):
        self.eval = eval
        self.transforms = self._eval_transf if self.eval else self._train_transf


class AugmentedConcatDataset(ConcatDataset):
    def __init__(self, *args, **kwargs):
        super(AugmentedConcatDataset, self).__init__(*args, **kwargs)
        self.eval_indices = set()

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        self.datasets[dataset_idx].set_eval(idx in self.eval_indices)
        return self.datasets[dataset_idx][sample_idx]

    def split(self, *ratios, eval=tuple()):
        assert np.isclose(np.sum(ratios), 1.0), 'the ratios do not sum to one'

        total = len(self)
        lengths = []
        for i, r in enumerate(ratios):
            n = round(total * r)
            lengths.append((total-np.sum(lengths)) if len(ratios)-1 == i else n)

        datasets = random_split(self, lengths)
        for i in eval:
            self.eval_indices.update(datasets[i].indices)

        return datasets

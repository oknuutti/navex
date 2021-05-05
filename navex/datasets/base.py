import bisect
import math
import os
import re
import random

import numpy as np
import PIL

import torch
from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataset import random_split
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms as tr

from .transforms import RandomDarkNoise, RandomExposure, PhotometricTransform, ComposedTransforms, \
    GeneralTransform, PairCenterCrop, PairRandomCrop, PairedIdentityTransform, RandomHomography, IdentityTransform, \
    RandomTiltWrapper, PairRandomScale, PairScaleToRange, RandomScale, ScaleToRange, GaussianNoise, \
    PairRandomHorizontalFlip, Clamp, UniformNoise, MatchChannels

from .tools import find_files_recurse, load_aflow

from .. import RND_SEED

RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
GRAY_MEAN, GRAY_STD = [0.449], [0.226]


def worker_init_fn(id):
    random.seed(RND_SEED)
    np.random.seed(RND_SEED)
    torch.random.manual_seed(RND_SEED)


class DataLoadingException(Exception):
    pass


class ImagePairDataset(VisionDataset):
    def __init__(self, root, aflow_loader=load_aflow, image_loader=default_loader, transforms=None):
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

            (img1, img2), aflow = self.preprocess(idx, (img1, img2), aflow)

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

    def preprocess(self, idx, imgs, aflow):
        return imgs, aflow


class PairIndexFileLoader:
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
    img0 path, img1 path, aflow path

    Args:
        index_file (string): path of the index file
        image_loader (callable): A function to load an image given its path.
        aflow_loader (callable): A function to load an absolute flow file given its path.
        transforms (callable, optional): A function/transform that takes in
            a pair of images and corresponding absolute flow and returns a transformed version of them
            E.g, ``RandomExposure`` which doesn't have a geometric distortion.

     Attributes:
        samples (list): List of ((img0 path, img1 path), aflow path) tuples
    """
    def __init__(self, index_file, aflow_loader, image_loader=default_loader,
                 index_file_loader=PairIndexFileLoader(), transforms=None):
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
        transforms = self.transforms if transforms is None else transforms
        super(SynthesizedPairDataset, self).__init__(root, transforms=transforms)

        # fv = AugmentedPairDatasetMixin.TR_NORM_RGB.mean if self.rgb else AugmentedPairDatasetMixin.TR_NORM_MONO.mean
        fv = np.nan
        self.warping_transforms = tr.Compose([
            IdentityTransform() if self.rgb else tr.Grayscale(num_output_channels=1),
            RandomHomography(max_tr=max_tr, max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                             min_size=min_size, fill_value=fv),
#            RandomTiltWrapper(magnitude=0.5),
        ])

        self.image_loader = image_loader
        self.samples = self._load_samples()

    def __getitem__(self, idx):
        img_pth = self.samples[idx]

        try:
            img1 = self.image_loader(img_pth)
            mask = self.valid_area(img1)

            eval = getattr(self, 'eval', False)
            if eval:
                with RandomSeed(idx):
                    img2, aflow = self.warping_transforms(img1)
            else:
                img2, aflow = self.warping_transforms(img1)

            aflow = aflow.reshape((-1, 2))
            aflow[np.logical_not(mask).flatten(), :] = np.nan
            aflow = aflow.reshape((*mask.shape, 2))
            if self.transforms is not None:
                (img1, img2), aflow = self.transforms((img1, img2), aflow)

        except Exception as e:
            raise DataLoadingException("Problem with dataset %s, index %s: %s" %
                                       (self.__class__, idx, self.samples[idx],)) from e

        return (img1, img2), aflow

    def valid_area(self, img):
        return np.ones(np.flip(img.size), dtype=bool)

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        raise NotImplemented()


class AugmentedPairDatasetMixin:
    TR_NORM_RGB = tr.Normalize(mean=RGB_MEAN, std=RGB_STD)
    TR_NORM_MONO = tr.Normalize(mean=GRAY_MEAN, std=GRAY_STD)

    def __init__(self, noise_max=0.1, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4), blind_crop=False,
                 margin=16, resize_max_size=1024, resize_max_sc=2.0, fill_value=None, eval=False, rgb=False):
        self.noise_max = noise_max
        self.rnd_gain = rnd_gain if isinstance(rnd_gain, (tuple, list)) else (1 / rnd_gain, rnd_gain)
        self.image_size = image_size
        self.max_sc = max_sc
        self.blind_crop = blind_crop
        self.eval = eval
        self.rgb = rgb
        self.margin = margin
        self.resize_max_size = resize_max_size
        self.resize_max_sc = resize_max_sc
        self.fill_value = fill_value
        if fill_value is None:
            self.fill_value = AugmentedPairDatasetMixin.TR_NORM_RGB.mean if self.rgb else AugmentedPairDatasetMixin.TR_NORM_MONO.mean
        self._init_transf()

    def _init_transf(self):
        self._train_transf = ComposedTransforms([
            PhotometricTransform(tr.Grayscale(num_output_channels=1)) if not self.rgb else PairedIdentityTransform(),
            PairRandomScale(min_size=max(self.image_size, 256), max_size=self.resize_max_size, max_sc=self.resize_max_sc)
                if self.resize_max_sc is not None and self.resize_max_sc >= 1 else PairedIdentityTransform(),
            PairRandomCrop(self.image_size, margin=self.margin, max_sc_diff=self.max_sc, blind_crop=self.blind_crop,
                           fill_value=self.fill_value),
            PairRandomHorizontalFlip(),
            GeneralTransform(tr.ToTensor()),
            PhotometricTransform(UniformNoise(self.noise_max), skip_1st=True),    # TODO: comment
            # PhotometricTransform(RandomDarkNoise(0, self.noise_max, 0.008, 3)),  # TODO: uncomment
            # TODO: config access to all color jitter params, NOTE: 0.1 hue jitter might remove rgb vs gray advantage
            PhotometricTransform(tr.ColorJitter(tuple(np.array(self.rnd_gain) - 1)[-1], 0.2, 0.2, 0.1), skip_1st=True)
                if self.rgb else PhotometricTransform(RandomExposure(*self.rnd_gain), skip_1st=True),
            PhotometricTransform(Clamp(0, 1)),
            PhotometricTransform(self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB),
        ])
        self._eval_transf = ComposedTransforms([
            PhotometricTransform(tr.Grayscale(num_output_channels=1)) if not self.rgb else PairedIdentityTransform(),
            PairScaleToRange(min_size=max(self.image_size, 256), max_size=self.resize_max_size, max_sc=self.resize_max_sc)
                if self.resize_max_sc is not None and self.resize_max_sc >= 1 else PairedIdentityTransform(),
            PairCenterCrop(self.image_size, margin=self.margin, max_sc_diff=self.max_sc, blind_crop=self.blind_crop,
                           fill_value=self.fill_value),
            GeneralTransform(tr.ToTensor()),
            PhotometricTransform(Clamp(0, 1)),
            PhotometricTransform(self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB),
        ])
        self.transforms = self._eval_transf if self.eval else self._train_transf

    def set_eval(self, eval):
        self.eval = eval
        self.transforms = self._eval_transf if self.eval else self._train_transf


class AugmentedDatasetMixin(AugmentedPairDatasetMixin):
    def __init__(self, noise_max, rnd_gain, image_size, max_tr, max_rot, max_shear, max_proj, min_size,
                 student_noise_sd, student_rnd_gain, resize_max_size=1024, resize_max_sc=2.0, eval=False, rgb=False):

        self.max_tr = max_tr
        self.max_rot = max_rot
        self.max_shear = max_shear
        self.max_proj = max_proj
        self.min_size = min_size
        self.student_noise_sd = student_noise_sd
        self.student_rnd_gain = student_rnd_gain if isinstance(student_rnd_gain, (tuple, list)) else \
                                (1 / student_rnd_gain, student_rnd_gain)

        super(AugmentedDatasetMixin, self).__init__(noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                                    resize_max_size=resize_max_size, resize_max_sc=resize_max_sc,
                                                    eval=eval, rgb=rgb)

    def _init_transf(self):
        self._train_transf = [
            tr.Compose([
                IdentityTransform() if self.rgb else tr.Grayscale(num_output_channels=1),
                RandomHomography(max_tr=self.max_tr, max_rot=self.max_rot, max_shear=self.max_shear, max_proj=self.max_proj,
                                 min_size=self.min_size, fill_value=np.nan, image_only=True),
                RandomScale(min_size=max(self.image_size, 256), max_size=self.resize_max_size, max_sc=self.resize_max_sc),
                tr.RandomCrop(self.image_size),
                tr.ToTensor(),
                tr.RandomHorizontalFlip(),
                UniformNoise(0.1),
                # RandomDarkNoise(0, self.noise_max, 0.008, 3),
                # TODO: config access to all color jitter params
                tr.ColorJitter(tuple(np.array(self.rnd_gain) - 1)[-1], 0.2, 0.2, 0.1)
                    if self.rgb else RandomExposure(*self.rnd_gain),
            ]),
            tr.Compose([
                # TODO: config access to all color jitter params
                tr.ColorJitter(tuple(np.array(self.student_rnd_gain) - 1)[-1], 0.2, 0.2, 0.1)
                    if self.rgb else RandomExposure(*self.student_rnd_gain),
                GaussianNoise(self.student_noise_sd),
            ]),
            tr.Compose([
                Clamp(0, 1),
                self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB,
            ]),
        ]

        self._eval_transf = [
            tr.Compose([
                IdentityTransform() if self.rgb else tr.Grayscale(num_output_channels=1),
                ScaleToRange(min_size=max(self.image_size, 256), max_size=np.inf, max_sc=np.inf),
                tr.CenterCrop(self.image_size),
                tr.ToTensor()]), 
            IdentityTransform(), 
            tr.Compose([
                Clamp(0, 1),
                self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB,
            ]),
        ]

        self.transforms = self._eval_transf if self.eval else self._train_transf


class BasicDataset(VisionDataset, AugmentedDatasetMixin):
    def __init__(self, root='data', folder=None, max_tr=0, max_rot=math.radians(15), max_shear=0.2, max_proj=0.8,
                 noise_max=0.20, rnd_gain=(0.5, 2), student_noise_sd=0.05, student_rnd_gain=(0.8, 1.2), image_size=512,
                 eval=False, rgb=False, npy=False, ext='.jpg', test=None, folder_depth=0):
        assert not npy, '.npy format not supported'
        self.npy = npy
        self.ext = ext
        self.test = test
        self.folder_depth = folder_depth

        AugmentedDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                       max_tr=max_tr, max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                                       student_noise_sd=student_noise_sd, student_rnd_gain=student_rnd_gain,
                                       min_size=image_size // 2, eval=eval, rgb=rgb)

        VisionDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms)

        self.image_loader = default_loader
        self.samples = self._load_samples()

    def _load_samples(self):
        return find_files_recurse(self.root, self.npy, self.ext, self.test, self.folder_depth)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_pth = self.samples[idx]

        try:
            img = self.image_loader(img_pth)
            img = self.transforms[0](img)
            noisy_img = self.transforms[1](img)
            # if not self.eval:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(torch.vstack((aug_img, noisy_img)).permute((1,2,0)).detach().cpu().numpy())
            #     print('sleep here')
            img, noisy_img = map(self.transforms[2], (img, noisy_img))
        except Exception as e:
            raise DataLoadingException("Problem with dataset %s, index %s: %s" %
                                       (self.__class__, idx, self.samples[idx],)) from e
        return img, noisy_img


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

        if hasattr(self.datasets[dataset_idx], 'set_eval'):
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


class ShuffledDataset(Subset):
    def __init__(self, dataset):
        indices = torch.randperm(len(dataset), generator=torch.default_generator).tolist()
        super(ShuffledDataset, self).__init__(dataset, indices=indices)


def split_tiered_data(primary, secondary, r_trn, r_val, r_tst):
    """
    Merge and then split different types of datasets for training, validation and testing
    :param primary: first class datasets, usable for training, validation and testing
    :param secondary: second class datasets, don't use for validation
    :param r_trn: ratio for training
    :param r_val: ratio for validation
    :param r_tst: ratio for testing
    :return: training, validation and testing datasets
    """
    ds1 = AugmentedConcatDataset(primary)
    ds2 = AugmentedConcatDataset(secondary)

    tot = len(ds1) + len(ds2)
    r0, r1, r2 = r_trn, r_val, r_tst

    p1 = int(r1 * tot) / len(ds1)
    p0 = (1 - p1) * r0 / (r0 + r2)
    p2 = (1 - p0 - p1,) if r_tst > 0 else tuple()
    s0 = (r0 * tot - p0 * len(ds1)) / len(ds2)
    s2 = (1 - s0,) if r_tst > 0 else tuple()

    ds1_split = ds1.split(p0, p1, *p2, eval=(1,) + ((2,) if r_tst > 0 else tuple()))
    ds2_split = ds2.split(s0, 0, s2, eval=(1, 2)) if r_tst > 0 else [ds2]

    trn = ShuffledDataset(AugmentedConcatDataset([ds1_split[0], ds2_split[0]]))
    val = ds1_split[1]
    tst = None if r_tst == 0 else ShuffledDataset(AugmentedConcatDataset([ds1_split[2], ds2_split[2]]))
    return trn, val, tst


class ExtractionImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, eval=True, rgb=False):
        if eval:
            self.transforms = tr.Compose([
                MatchChannels(rgb),
                tr.ToTensor(),
                tr.Normalize(mean=RGB_MEAN, std=RGB_STD) if rgb else
                tr.Normalize(mean=GRAY_MEAN, std=GRAY_STD),
            ])
        else:
            assert False, 'not implemented'

        # load samples
        exts = ('.jpg', '.png', '.bmp', '.jpeg', '.ppm')
        if os.path.isdir(path):
            self.samples = find_files_recurse(path, ext=exts)
            self.samples = sorted(self.samples, key=lambda x: tuple(map(int, re.findall(r'\d+', x))))
        elif path[-4:] in exts or path[-5:] == '.jpeg':
            self.samples = [path]
        else:
            with open(path) as fh:
                path = os.path.dirname(path)
                self.samples = list(map(lambda x: os.path.join(path, x.strip()), fh))

    @staticmethod
    def tensor2img(data):
        if data.dim() == 3:
            data = data[None, :, :, :]
        B, D, H, W = data.shape

        imgs = []
        for i in range(B):
            img = data[i, :, :, :].permute((1, 2, 0)).cpu().numpy()
            if D == 3:
                img = img * np.array(RGB_STD, dtype=np.float32) + np.array(RGB_MEAN, dtype=np.float32)
            else:
                img = img * np.array(GRAY_STD, dtype=np.float32) + np.array(GRAY_MEAN, dtype=np.float32)
            imgs.append(np.clip(img*255, 0, 255).astype(np.uint8))

        return imgs

    def __getitem__(self, idx):
        try:
            img = PIL.Image.open(self.samples[idx])
            if self.transforms is not None:
                img = self.transforms(img)
        except DataLoadingException as e:
            raise DataLoadingException("Problem with idx %s:\n%s" % (idx, self.samples[idx],)) from e

        return img

    def __len__(self):
        return len(self.samples)

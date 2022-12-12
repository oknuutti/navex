import bisect
import math
import os
import re
import random

import numpy as np
import quaternion
import PIL
import cv2

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

from .tools import ImageDB, find_files, find_files_recurse, load_aflow, if_none_q, q_times_v, normalize_v, eul_to_q

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
        (img1_pth, img2_pth), aflow_pth, *meta = self.samples[idx]

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

        return (img1, img2), aflow, *meta

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        raise NotImplemented()

    def preprocess(self, idx, imgs, aflow):
        return imgs, aflow


class DatabaseImagePairDataset(ImagePairDataset):
    def __init__(self, *args, **kwargs):
        self.indices, self.index = None, None
        super(DatabaseImagePairDataset, self).__init__(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['index']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        dbfile = os.path.join(self.root, 'dataset_all.sqlite')
        self.index = ImageDB(dbfile) if os.path.exists(dbfile) else None

    def _load_samples(self):
        dbfile = os.path.join(self.root, 'dataset_all.sqlite')
        self.index = ImageDB(dbfile) if os.path.exists(dbfile) else None

        index = {}
        for id, file, sc_sun_x, sc_sun_y, sc_sun_z, sc_qw, sc_qx, sc_qy, sc_qz, img_angle, \
                      sc_trg_x, sc_trg_y, sc_trg_z, trg_qw, trg_qx, trg_qy, trg_qz in self.index.get_all((
                          'id', 'file', 'sc_sun_x', 'sc_sun_y', 'sc_sun_z', 'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz',
                          'img_angle', 'sc_trg_x', 'sc_trg_y', 'sc_trg_z', 'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz')):
            index[id] = dict(file=file, sc_sun_v=np.array([sc_sun_x, sc_sun_y, sc_sun_z]),
                                        sc_q=if_none_q(sc_qw, sc_qx, sc_qy, sc_qz, fallback=quaternion.one)
                                             * eul_to_q((img_angle or 0,), 'z'),  # in opencv frame: +z cam axis, -y up
                                        sc_trg_v=np.array([sc_trg_x, sc_trg_y, sc_trg_z]),
                                        trg_q=if_none_q(trg_qw, trg_qx, trg_qy, trg_qz, fallback=np.quaternion(*[np.nan]*4)))

        aflow = find_files(os.path.join(self.root, 'aflow'), ext='.png')

        get_id = lambda f, i: int(f.split(os.path.sep)[-1].split('.')[0].split('_')[i])
        aflow = [f for f in aflow if get_id(f, 0) in index and get_id(f, 1) in index]
        self.indices = [(get_id(f, 0), get_id(f, 1)) for f in aflow]
        imgs = [(os.path.join(self.root, index[i]['file']),
                 os.path.join(self.root, index[j]['file'])) for i, j in self.indices]
        rel_q = [((index[i]['sc_q'].conj() * index[i]['trg_q']).conj()
                 * (index[j]['sc_q'].conj() * index[j]['trg_q'])).components for i, j in self.indices]
        rel_dist = [np.linalg.norm(index[j]['sc_trg_v'])
                    / np.linalg.norm(index[i]['sc_trg_v']) for i, j in self.indices]
        light1 = [np.ones((3,))*np.nan if index[i]['sc_q'] == quaternion.one else
                  q_times_v(index[i]['sc_q'].conj(), -normalize_v(index[i]['sc_sun_v'].astype(float))) for i, j in self.indices]
        light2 = [np.ones((3,))*np.nan if index[j]['sc_q'] == quaternion.one else
                  q_times_v(index[j]['sc_q'].conj(), -normalize_v(index[j]['sc_sun_v'].astype(float))) for i, j in self.indices]

        meta = (rel_q, rel_dist, light1, light2)    # NOTE: needs to be mirrored in SynthesizedPairDataset
        samples = list(zip(imgs, aflow, *meta))
        return samples


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
    def __init__(self, root, max_tr, max_rot, max_shear, max_proj, min_size, transforms=None, warp_crop=True,
                 image_loader=default_loader):
        transforms = self.transforms if transforms is None else transforms
        super(SynthesizedPairDataset, self).__init__(root, transforms=transforms)

        # fv = AugmentedPairDatasetMixin.TR_NORM_RGB.mean if self.rgb else AugmentedPairDatasetMixin.TR_NORM_MONO.mean
        fv = np.nan
        self.warping_transforms = tr.Compose([
            IdentityTransform() if self.rgb else tr.Grayscale(num_output_channels=1),
#            ScaleToRange(min_size=max(min_size, 256), max_size=getattr(self, 'resize_max_size', np.inf), max_sc=1.0),
            RandomHomography(max_tr=max_tr, max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                             min_size=min_size, fill_value=fv, crop_valid=warp_crop),
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

        # NOTE: needs to be mirrored in DatabaseImagePairDataset
        meta = (np.ones(4)*np.nan, np.nan, np.ones(3)*np.nan, np.ones(3)*np.nan)
        return (img1, img2), aflow, *meta

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
        self._upd_tranf()

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
        self._test_transf = ComposedTransforms([
            PhotometricTransform(tr.Grayscale(num_output_channels=1)) if not self.rgb else PairedIdentityTransform(),
            GeneralTransform(tr.ToTensor()),
            PhotometricTransform(Clamp(0, 1)),
            PhotometricTransform(self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB),
        ])

    def _upd_tranf(self):
        self.transforms = {False: self._train_transf, True: self._eval_transf, 'test': self._test_transf}[self.eval]

    def set_eval(self, eval):
        self.eval = eval
        self._upd_tranf()


class AugmentedDatasetMixin(AugmentedPairDatasetMixin):
    def __init__(self, noise_max, rnd_gain, image_size, max_tr, max_rot, max_shear, max_proj, min_size,
                 student_noise_sd, student_rnd_gain, resize_max_size=1024, resize_max_sc=2.0,
                 warp_crop=True, eval=False, rgb=False):

        self.max_tr = max_tr
        self.max_rot = max_rot
        self.max_shear = max_shear
        self.max_proj = max_proj
        self.min_size = min_size
        self.warp_crop = warp_crop
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
                                 min_size=self.min_size, fill_value=np.nan, crop_valid=self.warp_crop, image_only=True),
                RandomScale(min_size=max(self.image_size, 256), max_size=self.resize_max_size, max_sc=self.resize_max_sc),
                tr.RandomCrop(self.image_size),
                tr.ToTensor(),
                tr.RandomHorizontalFlip(),
                UniformNoise(self.noise_max),
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

        self._test_transf =tr.Compose([
            tr.Compose([
                IdentityTransform() if self.rgb else tr.Grayscale(num_output_channels=1),
                tr.ToTensor()]),
            IdentityTransform(),
            tr.Compose([
                Clamp(0, 1),
                self.TR_NORM_MONO if not self.rgb else self.TR_NORM_RGB,
            ]),
        ])


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
    :param secondary: second class datasets, use only for training
    :param r_trn: ratio of primary data used for training
    :param r_val: ratio of primary data used for validation
    :param r_tst: ratio of primary data used for testing
    :return: training, validation and testing datasets
    """
    ds1 = AugmentedConcatDataset(primary)
    ds2 = AugmentedConcatDataset(secondary) if len(secondary) > 0 else []

    p_tst = (1 - r_trn - r_val,) if r_tst > 0 else tuple()
    ds1_split = ds1.split(r_trn, r_val, *p_tst, eval=(1,) + ((2,) if r_tst > 0 else tuple()))

    trn = ShuffledDataset(AugmentedConcatDataset([ds1_split[0], ds2])) if len(ds2) > 0 else ds1_split[0]
    val = ds1_split[1]
    tst = None if r_tst == 0 else ds1_split[2]
    return trn, val, tst


class ExtractionImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, recurse=True, eval=True, rgb=False, regex=None):
        if eval:
            self.transforms = tr.Compose([
                MatchChannels(rgb),
                tr.ToTensor(),
                tr.Normalize(mean=RGB_MEAN, std=RGB_STD) if rgb else
                tr.Normalize(mean=GRAY_MEAN, std=GRAY_STD),
            ])
        else:
            assert False, 'not implemented'

        if not isinstance(paths, list):
            paths = [paths]

        self.samples = []
        skip_sort = False
        for path in paths:
            if isinstance(path, np.ndarray):
                # image already loaded
                self.samples.append(path)
                skip_sort = True
            else:
                # load samples
                exts = ('.jpg', '.png', '.bmp', '.jpeg', '.ppm') if regex is None else re.compile(regex)
                if os.path.isdir(path):
                    self.samples.extend(find_files_recurse(path, ext=exts, depth=100 if recurse else 0))
                elif isinstance(exts, re.Pattern) and re.match(exts, path) \
                        or isinstance(exts, tuple) and (path[-4:] in exts or path[-5:] == '.jpeg'):
                    self.samples.append(path)
                else:
                    with open(path) as fh:
                        path = os.path.dirname(path)
                        self.samples.extend(list(map(lambda x: os.path.join(path, x.strip()), fh)))

        if not skip_sort:
            self.samples = sorted(self.samples, key=lambda x: tuple(map(int, re.findall(r'\d+', x))))

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
        path = self.samples[idx]
        try:
            if 0:
                img = PIL.Image.open(self.samples[idx])
            else:
                if isinstance(path, str):
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                elif isinstance(path, np.ndarray):
                    img = path
                else:
                    assert False, 'wrong type of data stored in self.samples'

                if img.dtype == np.uint16:
                    from .tools import preprocess_image
                    img, _ = preprocess_image(img, gamma=1.8)
                img = PIL.Image.fromarray(img)

            if self.transforms is not None:
                img = self.transforms(img)
        except Exception as e:
            raise DataLoadingException("Problem with idx %s:\n%s" % (idx, path)) from e

        return img

    def __len__(self):
        return len(self.samples)

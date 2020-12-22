import logging
import random
import math
import os

import numpy as np
import cv2
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Lambda

from .. import RND_SEED


def worker_init_fn(id):
    random.seed(RND_SEED)
    np.random.seed(RND_SEED)
    torch.random.manual_seed(RND_SEED)


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
            aflow = self.aflow_loader(aflow_pth)

            if self.transforms is not None:
                (img1, img2), aflow = self.transforms((img1, img2), aflow)

        except DataLoadingException as e:
            raise DataLoadingException("Problem with idx %s:\n%s" % (idx, self.samples[idx],)) from e

        return (img1, img2), aflow

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        raise NotImplemented()


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


class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, images, aflow):
        return images, aflow


class PhotometricTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, aflow):
        return tuple(map(self.transform, images)), aflow


class GeneralTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, aflow):
        return tuple(map(self.transform, images)), self.transform(aflow)


class ComposedTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, aflow):
        for transform in self.transforms:
            images, aflow = transform(images, aflow)
        return images, aflow


class PairedRandomCrop:
    def __init__(self, shape, random=True, max_sc_diff=None, random_sc=True):
        self.shape = (shape, shape) if isinstance(shape, int) else shape       # yx i.e. similar to aflow.shape
        self.random = random
        self.max_sc_diff = max_sc_diff
        self.random_sc = random_sc

    def most_ok_in_window(self, mask, sc=4):
        n, m = np.array(self.shape) // sc
        c = 1 / m / n
        mask_sc = cv2.resize(mask.astype(np.float32), None, fx=1/sc, fy=1/sc, interpolation=cv2.INTER_AREA)

        res = cv2.filter2D(mask_sc, ddepth=cv2.CV_32F, anchor=(0, 0),
                           kernel=np.ones((n, m), dtype='float32') * c,
                           borderType=cv2.BORDER_ISOLATED)
        res[res.shape[0] - n:, :] = 0
        res[:, res.shape[1] - m:] = 0
        res[res < np.max(res) * 0.5] = 0  # only possible to select windows that are at most half as bad as best window

        a = np.cumsum(res.flatten())
        rnd_idx = np.argmax(a > np.random.uniform(0, a[-1]))
        bst_idx = np.argmax(res.flatten())

        rnd_idxs = np.array(np.unravel_index(rnd_idx, res.shape)) * sc
        bst_idxs = np.array(np.unravel_index(bst_idx, res.shape)) * sc
        return bst_idxs, rnd_idxs

    def __call__(self, imgs, aflow):
        img1, img2 = imgs
        n, m = self.shape
        mask = np.logical_not(np.isnan(aflow[:, :, 0]))
        bst_idxs, rnd_idxs = self.most_ok_in_window(mask)

        for t in range(2):
            if t == 1 or not self.random:
                j1, i1 = bst_idxs
                is_random = False
            else:
                j1, i1 = rnd_idxs
                is_random = True
            c_aflow = aflow[j1:j1+m, i1:i1+n]

            # if too few valid correspondences, pick the central crop instead
            ratio_valid = 1 - np.mean(np.isnan(c_aflow[:, :, 0]))
            if ratio_valid > 0.05:
                break

        if ratio_valid == 0:
            # if this becomes a real problem, use SafeDataset and SafeDataLoader from nonechucks, then return None here
            raise DataLoadingException("no valid correspondences even for central crop")

        c_img1 = img1.crop((i1, j1, i1+m, j1+n))
        c_mask = mask[j1:j1+m, i1:i1+n]

        # determine current scale of img2 relative to img1 based on aflow
        xy1 = np.stack(np.meshgrid(range(m), range(n)), axis=2).reshape((-1, 2))
        ic1, jc1 = np.mean(xy1[c_mask.flatten(), :], axis=0)
        sc1 = np.sqrt(np.mean(np.sum((xy1[c_mask.flatten(), :] - np.array((ic1, jc1)))**2, axis=1)))
        ic2, jc2 = np.nanmean(c_aflow, axis=(0, 1))
        sc2 = np.sqrt(np.nanmean(np.sum((c_aflow - np.array((ic2, jc2)))**2, axis=2)))
        curr_sc = sc2 / sc1

        # determine target scale based on current scale, self.max_sc_diff, and self.random_sc
        lsc = abs(np.log10(self.max_sc_diff))
        if self.random_sc and is_random:                       # if first try fails, don't scale for second try
            trg_sc = 10**np.random.uniform(-lsc, lsc)
        else:
            min_sc, max_sc = 10 ** (-lsc), 10 ** lsc
            trg_sc = np.clip(curr_sc, min_sc, max_sc)

        # scale aflow
        c_aflow = c_aflow * trg_sc/curr_sc

        # use cv2.filter2D and argmax for img2 also
        trg_full_shape = int(img2.size[1] * trg_sc/curr_sc), int(img2.size[0] * trg_sc/curr_sc)
        idxs = c_aflow.reshape((-1, 2))[np.logical_not(np.isnan(c_aflow[:, :, 0].flatten())), :].astype('uint16')
        idxs = idxs[np.logical_and(idxs[:, 0] < trg_full_shape[1], idxs[:, 1] < trg_full_shape[0]), :]
        c_ok = np.zeros(trg_full_shape, dtype='float32')
        c_ok[idxs[:, 1], idxs[:, 0]] = 1
        (j2, i2), _ = self.most_ok_in_window(c_ok)

        # massage aflow
        c_aflow = (c_aflow - np.array((i2, j2), dtype=c_aflow.dtype)).reshape((-1, 2))
        c_aflow[np.any(c_aflow < 0, axis=1), :] = np.nan
        c_aflow[np.logical_or(c_aflow[:, 0] > m - 1, c_aflow[:, 1] > n - 1), :] = np.nan
        c_aflow = c_aflow.reshape((n, m, 2))

        # crop and resize image 2
        i2s, j2s, i2e, j2e = (np.array((i2, j2, i2+m, j2+n))*curr_sc/trg_sc + 0.5).astype('uint16')
        c_img2 = img2.crop((i2s, j2s, i2e, j2e)).resize((m, n))

        if 0:
            import matplotlib.pyplot as plt
            plt.figure(1), plt.imshow(np.array(c_img1))
            plt.figure(2), plt.imshow(np.array(c_img2))
            for i in range(8):
                idx = np.argmax((1-np.isnan(c_aflow[:, :, 0].flatten()).astype(np.float32))
                                * np.random.lognormal(0, 1, (n*m,)))
                y0, x0 = np.unravel_index(idx, c_aflow.shape[:2])
                plt.figure(1), plt.plot(x0, y0, 'x')
                plt.figure(2), plt.plot(c_aflow[y0, x0, 0], c_aflow[y0, x0, 1], 'x')

            # plt.figure(3), plt.imshow(img1), plt.figure(4), plt.imshow(img2)
            # plt.figure(3), plt.imshow(c_mask), plt.figure(4), plt.imshow(c_ok.T[j2:j2+n, i2:i2+m])
            plt.show()
            min_i, min_j = np.nanmin(c_aflow, axis=(0, 1))
            max_i, max_j = np.nanmax(c_aflow, axis=(0, 1))
            assert min_i >= 0 and min_j >= 0, 'flow coord less than zero: i: %s, j: %s' % (min_i, min_j)
            assert max_i < m and max_j < n, 'flow coord greater than cropped size: i: %s, j: %s' % (max_i, max_j)

        assert tuple(c_img1.size) == tuple(np.flip(self.shape)), 'Image 1 is wrong size: %s' % (c_img1.size,)
        assert tuple(c_img2.size) == tuple(np.flip(self.shape)), 'Image 2 is wrong size: %s' % (c_img2.size,)
        assert tuple(c_aflow.shape[:2]) == tuple(self.shape), 'Absolute flow is wrong shape: %s' % (c_aflow.shape,)

        return (c_img1, c_img2), c_aflow


class PairedCenterCrop(PairedRandomCrop):
    def __init__(self, shape, max_sc_diff=None):
        super(PairedCenterCrop, self).__init__(shape, random=0, max_sc_diff=max_sc_diff, random_sc=False)

    def __call__(self, imgs, aflow):
        return super(PairedCenterCrop, self).__call__(imgs, aflow)


class DataLoadingException(Exception):
    pass


class RandomExposure:
    def __init__(self, min_gain, max_gain):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, img):
        """
        Args:
            img (Tensor): Input image as a Tensor.

        Returns:
            transformed image: image with random exposure adjustment (gain).
        """
        gain = math.exp(random.uniform(math.log(self.min_gain), math.log(self.max_gain)))
        transform = Lambda(lambda img: torch.clamp(img * gain, 0, 1))
        img_t = transform(img)
        # plt.imshow(img_t.detach().cpu().numpy().squeeze())
        # plt.show()
        return img_t

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'min_gain={0}'.format(self.min_gain)
        format_string += ', max_gain={0})'.format(self.max_gain)
        return format_string


class RandomDarkNoise:
    def __init__(self, min_level, max_level, gain=0.1, pow=3):
        self.min_level = min_level
        self.max_level = max_level
        self.gain = gain
        self.pow = pow

    def __call__(self, img):
        """
        Args:
            img (Tensor): Input image as a Tensor.

        Returns:
            transformed image: image with random exposure adjustment (gain).
        """
        mean = random.uniform(self.min_level**(1/self.pow), self.max_level**(1/self.pow)) ** self.pow
        sd = math.sqrt(self.gain * mean)   # dark shot noise (i.e. photon noise of dark current)
        transform = Lambda(lambda img: torch.clamp(img + mean + sd*torch.randn_like(img), 0, 1))
        img_t = transform(img)
        return img_t

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'min_level={0}'.format(self.min_level)
        format_string += ', max_level={0}'.format(self.max_level)
        format_string += ', pow={0})'.format(self.pow)
        return format_string

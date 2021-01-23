import math
import random

import numpy as np
import scipy.interpolate as interp
import PIL
import cv2
import torch
from torchvision.transforms import Lambda


class GeneralTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, aflow):
        return tuple(map(self.transform, images)), self.transform(aflow)


class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        return data


class PairedIdentityTransform:
    def __init__(self):
        pass

    def __call__(self, images, aflow):
        return images, aflow


class PhotometricTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, aflow):
        return tuple(map(self.transform, images)), aflow


class ComposedTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, aflow):
        for transform in self.transforms:
            images, aflow = transform(images, aflow)
        return images, aflow


class PairedRandomCrop:
    def __init__(self, shape, random=True, max_sc_diff=None, random_sc=True, blind_crop=False):
        self.shape = (shape, shape) if isinstance(shape, int) else shape       # yx i.e. similar to aflow.shape
        self.random = random
        self.max_sc_diff = max_sc_diff
        self.random_sc = random_sc
        self.blind_crop = blind_crop  # don't try to validate cropping location, good for certain datasets

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

        if self.blind_crop:
            bst_idxs = (img1.size[1] - n) // 2, (img1.size[0] - m) // 2   # center crop
            rnd_idxs = int(random.uniform(0, img1.size[1] - n)), int(random.uniform(0, img1.size[0] - m))
        else:
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
            from navex.datasets.base import DataLoadingException
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

        if self.blind_crop:
            i2, j2 = np.nanmean(c_aflow, axis=(0, 1)).astype(np.int) - np.array([m//2, n//2], dtype=np.int)
        else:
            # use cv2.filter2D and argmax for img2 also
            trg_full_shape = int(img2.size[1] * trg_sc / curr_sc), int(img2.size[0] * trg_sc / curr_sc)
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
        try:
            c_img2 = img2.crop((i2s, j2s, i2e, j2e)).resize((m, n))
        except Exception as e:
            print('weird problem')

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
    def __init__(self, shape, max_sc_diff=None, blind_crop=False):
        super(PairedCenterCrop, self).__init__(shape, random=0, max_sc_diff=max_sc_diff,
                                               random_sc=False, blind_crop=blind_crop)

    def __call__(self, imgs, aflow):
        return super(PairedCenterCrop, self).__call__(imgs, aflow)


class RandomHomography:
    def __init__(self, max_tr, max_rot, max_shear, max_proj, fill_value=0):
        self.max_tr = max_tr
        self.max_rot = max_rot
        self.max_shear = max_shear
        self.max_proj = max_proj
        self.fill_value = fill_value

    def random_H(self, w, h):
        tr_x = random.uniform(-self.max_tr, self.max_tr) * w
        tr_y = random.uniform(-self.max_tr, self.max_tr) * h
        rot = random.uniform(-self.max_rot, self.max_rot)
        He = np.array([[math.cos(rot), -math.sin(rot), tr_x],
                       [math.sin(rot), math.cos(rot),  tr_y],
                       [0,             0,              1]], dtype=np.float32)

        sh_x = random.uniform(-self.max_shear, self.max_shear)
        sh_y = random.uniform(-self.max_shear, self.max_shear)
        Ha = np.array([[1, sh_x, 0],
                       [sh_y, 1, 0],
                       [0,    0, 1]], dtype=np.float32)

        p1 = random.uniform(-self.max_proj, self.max_proj) / w
        p2 = random.uniform(-self.max_proj, self.max_proj) / h
        Hp = np.array([[1,  0,  0],
                       [0,  1,  0],
                       [p1, p2, 1]], dtype=np.float32)

        H = np.array([[1, 0, w/2],
                      [0, 1, h/2],
                      [0, 0,    1]], dtype=np.float32) \
            .dot(He).dot(Ha).dot(Hp) \
            .dot(np.array([[1,  0,  -w/2],
                           [0,  1,  -h/2],
                           [0,  0,  1]], dtype=np.float32))
        return H

    def __call__(self, img):
        # from https://stackoverflow.com/questions/16682965/how-to-generaterandomtransform-with-opencv
        #  - NOTE: not certain if actually correct
        from .base import unit_aflow
        w, h = img.size

        H = self.random_H(w, h)

        aflow_shape = (h, w, 2)
        uh_aflow = np.concatenate((unit_aflow(w, h), np.ones((*aflow_shape[:2], 1), dtype=np.float32)), axis=2)
        w_aflow = uh_aflow.reshape((-1, 3)).dot(H.T)
        w_aflow = (w_aflow[:, :2] / w_aflow[:, 2:]).reshape(aflow_shape)

        grid = uh_aflow.reshape((-1, 3)).dot(np.linalg.inv(H.T))
        grid = (grid[:, :2] / grid[:, 2:]).reshape(aflow_shape)

        ifun = interp.RegularGridInterpolator((np.arange(h), np.arange(w)), np.array(img),
                                              fill_value=self.fill_value, bounds_error=False)
        w_img = PIL.Image.fromarray(ifun(np.flip(grid, axis=2)))

        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(np.array(w_img))
            plt.show()

        return w_img, w_aflow


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

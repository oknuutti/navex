import math
import random

import numpy as np
import scipy.interpolate as interp
import PIL
import cv2
import torch
from r2d2.tools.transforms import RandomTilt
from r2d2.tools.transforms_tools import persp_apply
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


class RandomScale:
    def __init__(self, min_size, max_size, max_sc, random=True, interp_method=PIL.Image.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        self.max_sc = max_sc
        self.random = random
        self.interp_method = interp_method    # or should use LANCZOS for better quality?

    def __call__(self, imgs, aflow):
        img1, img2 = imgs
        w, h = img1.size
        min_side = min(w, h)
        min_sc = self.min_size / min_side
        max_sc = min(self.max_sc, self.max_size / min_side)

        if self.random:
            trg_sc = math.exp(random.uniform(math.log(min_sc), math.log(max_sc)))
#            trg_sc = random.uniform(min_sc, max_sc)
        else:
            trg_sc = np.clip(1.0, min_sc, max_sc)

        if not np.isclose(trg_sc, 1.0):
            nw, nh = round(w * trg_sc), round(h * trg_sc)
            img1 = img1.resize((nw, nh), self.interp_method)
            aflow = cv2.resize(aflow, (nw, nh), interpolation=cv2.INTER_NEAREST)

        return (img1, img2), aflow


class ScaleToRange(RandomScale):
    def __init__(self, min_size, max_size, max_sc):
        super(ScaleToRange, self).__init__(min_size, max_size, max_sc, random=False)


class PairedRandomCrop:
    def __init__(self, shape, random=True, max_sc_diff=None, random_sc_diff=True, fill_value=None,
                 blind_crop=False, interp_method=PIL.Image.BILINEAR):
        self.shape = (shape, shape) if isinstance(shape, int) else shape       # yx i.e. similar to aflow.shape
        self.random = random
        self.max_sc_diff = max_sc_diff
        self.random_sc_diff = random_sc_diff
        self.fill_value = 0 if fill_value is None else (np.array(fill_value) * 255).reshape((1, 1, -1)).astype('uint8')
        self.blind_crop = blind_crop  # don't try to validate cropping location, good for certain datasets
        self.interp_method = interp_method

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

    def __call__(self, imgs, aflow, debug=False):
        img1, img2 = imgs
        n, m = self.shape

        if m > img1.size[0] or n > img1.size[1]:
            # need to pad as otherwise image too small
            img1, aflow = self._pad((m, n), img1, aflow, first=True)

        mask = np.logical_not(np.isnan(aflow[:, :, 0]))

        if self.blind_crop:
            # simplified, fast window selection
            bst_idxs = round((img1.size[1] - n) / 2), round((img1.size[0] - m) / 2)   # center crop
            rnd_idxs = round(random.uniform(0, img1.size[1] - n)), round(random.uniform(0, img1.size[0] - m))
        else:
            # secure window selection
            bst_idxs, rnd_idxs = self.most_ok_in_window(mask)

        assert np.all(np.array((*bst_idxs, *rnd_idxs)) >= 0), \
               'image size is smaller than the crop area: %s vs %s' % (img1.size, (m, n))

        for t in range(2):
            if t == 1 and self.blind_crop:
                # revert to secure window selection instead of the simplified one
                bst_idxs, rnd_idxs = self.most_ok_in_window(mask)
                j1, i1 = bst_idxs
                is_random = False
            elif t == 1 or not self.random:
                j1, i1 = bst_idxs
                is_random = False
            else:
                j1, i1 = rnd_idxs
                is_random = True
            c_aflow = aflow[j1:j1+m, i1:i1+n, :]

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

        # determine target scale based on current scale, self.max_sc_diff, and self.random_sc_diff
        lsc = abs(np.log10(self.max_sc_diff))
        if self.random_sc_diff and is_random:                       # if first try fails, don't scale for second try
            trg_sc = 10**np.random.uniform(-lsc, lsc)
        else:
            min_sc, max_sc = 10 ** (-lsc), 10 ** lsc
            trg_sc = np.clip(curr_sc, min_sc, max_sc)

        cm, cn = math.ceil(m * curr_sc / trg_sc), math.ceil(n * curr_sc / trg_sc)
        if cm > img2.size[0] or cn > img2.size[1]:
            # padding is necessary
            img2, c_aflow = self._pad((cm, cn), img2, c_aflow, first=False)

        # scale aflow
        c_aflow = c_aflow * trg_sc/curr_sc
        trg_full_shape = math.ceil(img2.size[1] * trg_sc / curr_sc), math.ceil(img2.size[0] * trg_sc / curr_sc)

        if self.blind_crop:
            i2, j2 = (np.nanmean(c_aflow, axis=(0, 1)) - np.array([m/2, n/2]) + 0.5).astype(np.int)
            i2, j2 = np.clip(i2, 0, trg_full_shape[1] - m), np.clip(j2, 0, trg_full_shape[0] - n)
        else:
            # use cv2.filter2D and argmax for img2 also
            idxs = c_aflow.reshape((-1, 2))[np.logical_not(np.isnan(c_aflow[:, :, 0].flatten())), :].astype(np.int)
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
        i2s, i2e, j2s, j2e = (np.array((i2, i2+m, j2, j2+n))*curr_sc/trg_sc + 0.5).astype(np.int)
        i2s, j2s = max(0, i2s), max(0, j2s)
        i2e, j2e = min(img2.size[0], i2e), min(img2.size[1], j2e)
        c_img2 = img2.crop((i2s, j2s, i2e, j2e)).resize((m, n), self.interp_method)

        if debug:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(np.array(c_img1))
            axs[1].imshow(np.array(c_img2))
            for i in range(8):
                idx = np.argmax((1-np.isnan(c_aflow[:, :, 0].flatten()).astype(np.float32))
                                * np.random.lognormal(0, 1, (n*m,)))
                y0, x0 = np.unravel_index(idx, c_aflow.shape[:2])
                axs[0].plot(x0, y0, 'x')
                axs[1].plot(c_aflow[y0, x0, 0], c_aflow[y0, x0, 1], 'x')

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

    def _pad(self, min_size, img, aflow, first):
        n_ch = len(img.getbands())
        w, h = img.size
        nw, nh = max(min_size[0], w), max(min_size[1], h)
        psi, psj = (nw - w) // 2, (nh - h) // 2
        img_arr = np.array(img)
        p_img = np.ones((nh, nw, n_ch), dtype=img_arr.dtype) * self.fill_value
        p_img[psj:psj + h, psi:psi + w, :] = np.atleast_3d(img_arr)
        p_img = PIL.Image.fromarray(p_img.squeeze())

        if first:
            p_aflow = np.ones((nh, nw, 2), dtype=np.float32) * np.nan
            p_aflow[psj:psj + h, psi:psi + w, :] = aflow
        else:
            p_aflow = aflow + np.array([psi, psj], dtype=np.float32)

        return p_img, p_aflow


class PairedCenterCrop(PairedRandomCrop):
    def __init__(self, shape, max_sc_diff=None, fill_value=None, blind_crop=False):
        super(PairedCenterCrop, self).__init__(shape, random=0, max_sc_diff=max_sc_diff,
                                               random_sc_diff=False, blind_crop=blind_crop, fill_value=fill_value)

    def __call__(self, imgs, aflow):
        return super(PairedCenterCrop, self).__call__(imgs, aflow)


class RandomHomography:
    def __init__(self, max_tr, max_rot, max_shear, max_proj, min_size, fill_value=np.nan):
        self.max_tr = max_tr
        self.max_rot = max_rot
        self.max_shear = max_shear
        self.max_proj = max_proj
        self.min_size = min_size
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

        H = He.dot(Ha).dot(Hp)
        return H

    def __call__(self, img):
        # from https://stackoverflow.com/questions/16682965/how-to-generaterandomtransform-with-opencv
        #  - NOTE: not certain if actually correct
        from .base import unit_aflow
        w, h = img.size
        aflow_shape = (h, w, 2)
        uh_aflow = np.concatenate((unit_aflow(w, h), np.ones((*aflow_shape[:2], 1), dtype=np.float32)), axis=2)

        ok = False
        bad_h = 0
        for i in range(5):
            H = self.random_H(w, h)
            w_aflow = uh_aflow.reshape((-1, 3)).dot(H.T)
            if np.any(np.isclose(w_aflow[:, 2:], 0)):
                bad_h += 1
                continue

            w_aflow = (w_aflow[:, :2] / w_aflow[:, 2:]).reshape(aflow_shape)
            corners = w_aflow[[0, 0, -1, -1], [0, -1, 0, -1], :]

            if np.any(np.isnan(self.fill_value)):
                # define resulting image so that not need to fill any values
                # TODO: FIX THIS: sometimes results in negative width or height
                x0 = max(corners[0, 0], corners[2, 0])
                x1 = min(corners[1, 0], corners[3, 0])
                y0 = max(corners[0, 1], corners[1, 1])
                y1 = min(corners[2, 1], corners[3, 1])
            else:
                # define resulting image so that whole transformed image included
                (x0, y0), (x1, y1) = np.min(corners, axis=0), np.max(corners, axis=0)

            nw, nh = math.ceil(x1 - x0), math.ceil(y1 - y0)
            if nw >= self.min_size and nh >= self.min_size:
                ok = True
                break

        assert ok, ('Failed to generate valid homography, '
                    'resulting new size %s is less than the required %d, source size was %s (or bad H? %d/5)') % (
                        (nw, nh), self.min_size, (w, h), bad_h)

        w_aflow -= np.array([x0, y0]).reshape((1, 1, 2))
        uh_grid = np.concatenate((unit_aflow(nw, nh) + np.array([[[x0, y0]]]),
                                  np.ones((nh, nw, 1), dtype=np.float32)), axis=2)
        grid = uh_grid.reshape((-1, 3)).dot(np.linalg.inv(H.T))
        grid = (grid[:, :2] / grid[:, 2:]).reshape((nh, nw, 2))

        ifun = interp.RegularGridInterpolator((np.arange(h), np.arange(w)), np.array(img), bounds_error=False,
                                              fill_value=np.array(self.fill_value)*255)
        img_arr = ifun(np.flip(grid, axis=2))
        w_img = PIL.Image.fromarray(img_arr.astype(np.uint8))

        return w_img, w_aflow


class RandomTiltWrapper(RandomTilt):
    def __call__(self, img):
        scaled_and_distorted_image = \
            super(RandomTiltWrapper, self).__call__(dict(img=img, persp=(1, 0, 0, 0, 1, 0, 0, 0)))

        W, H = img.size
        trf = scaled_and_distorted_image['persp']

        # compute optical flow
        xy = np.mgrid[0:H, 0:W][::-1].reshape((2, -1)).T
        aflow = persp_apply(trf, xy).astype(np.float32)

        aflow[np.any(aflow < 0, axis=1), :] = np.nan
        aflow[np.logical_or(aflow[:, 0] > W - 1, aflow[:, 1] > H - 1), :] = np.nan
        aflow = aflow.reshape((H, W, 2))

        return scaled_and_distorted_image['img'], aflow


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

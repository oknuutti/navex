import math
import random

import numpy as np
import scipy.interpolate as interp
import PIL
import cv2
import torch
from PIL import ImageOps
import torchvision.transforms as tr

from ..models import tools
from .tools import unit_aflow, show_pair


class GeneralTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, images, aflow, *meta):
        return tuple(map(self.transform, images)), self.transform(aflow), *meta


class IdentityTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        return data


class PairedIdentityTransform:
    def __init__(self):
        pass

    def __call__(self, images, aflow, *meta):
        return images, aflow, *meta


class PhotometricTransform:
    def __init__(self, transform, single=False, skip_1st=False):
        self.transform = transform
        self.single = single
        self.skip_1st = skip_1st

    def __call__(self, images, *aflow_meta):
        if len(aflow_meta) == 0:
            (images, aflow), meta = images, tuple()
        else:
            aflow, meta = aflow_meta[0], aflow_meta[1:]
        return (self.transform(images) if self.single else (
                   (images[0], *map(self.transform, images[1:])) if self.skip_1st else
                   tuple(map(self.transform, images))
               )), aflow, *meta


class ComposedTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, aflow, *meta):
        for transform in self.transforms:
            images, aflow, *meta = transform(images, aflow, *meta)
        return images, aflow, *meta


class MatchChannels:
    def __init__(self, rgb):
        self.rgb = rgb
        self.tr = tr.Grayscale(num_output_channels=3 if self.rgb else 1)

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            d = img.shape[0 if img.dim() == 3 else 1]
        else:
            d = len(img.getbands())

        if self.rgb and d >= 3 or not self.rgb and d == 1:
            return img
        return self.tr(img)


class RandomScale:
    def __init__(self, min_size, max_size, max_sc, random=True, interp_method=PIL.Image.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        self.max_sc = max_sc
        self.random = random
        self.interp_method = interp_method    # or should use LANCZOS for better quality?

    def __call__(self, img):
        w, h = img.size
        min_side = min(w, h)
        min_sc = self.min_size / min_side
        max_sc = max(min_sc, min(self.max_sc, self.max_size / min_side))

        if self.random:
            trg_sc = math.exp(random.uniform(math.log(min_sc), math.log(max_sc)))
#            trg_sc = random.uniform(min_sc, max_sc)
        else:
            trg_sc = np.clip(1.0, min_sc, max_sc)

        if not np.isclose(trg_sc, 1.0):
            nw, nh = round(w * trg_sc), round(h * trg_sc)
            img = img.resize((nw, nh), self.interp_method)

        return img


class PairRandomScale(RandomScale):
    def __init__(self, *args, resize_img2=False, **kwargs):
        super(PairRandomScale, self).__init__(*args, **kwargs)
        self.resize_img2 = resize_img2

    def __call__(self, imgs, aflow, *meta):
        img1, img2 = imgs
        w, h = img1.size
        img1 = super(PairRandomScale, self).__call__(img1)
        nw, nh = img1.size

        if w != nw or h != nh:
            aflow = cv2.resize(aflow, (nw, nh), interpolation=cv2.INTER_NEAREST)

        if self.resize_img2:
            w, h = img2.size
            img2 = super(PairRandomScale, self).__call__(img2)
            nw, nh = img2.size

            if w != nw or h != nh:
                aflow *= nw / w

        return (img1, img2), aflow, *meta


class ScaleToRange(RandomScale):
    def __init__(self, *args, **kwargs):
        super(ScaleToRange, self).__init__(*args, random=False, **kwargs)


class PairScaleToRange(PairRandomScale):
    def __init__(self, *args, **kwargs):
        super(PairScaleToRange, self).__init__(*args, random=False, **kwargs)


class PairRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, aflow, *meta):
        img1, img2 = imgs
        w2, h2 = img2.size

        if random.uniform(0, 1) < self.p:
            img1 = ImageOps.mirror(img1)
            img2 = ImageOps.mirror(img2)
            aflow = np.fliplr(aflow).copy()
            aflow[:, :, 0] = w2 - aflow[:, :, 0]

        return (img1, img2), aflow, *meta


class PairRandomCrop:
    def __init__(self, shape, random=True, max_sc_diff=None, random_sc_diff=True, fill_value=None,
                 margin=16, blind_crop=False, interp_method=PIL.Image.BILINEAR, only_crop=False, return_bounds=False):
        self.shape = (shape, shape) if isinstance(shape, int) else shape       # yx i.e. similar to aflow.shape
        self.random = random
        self.max_sc_diff = max_sc_diff
        self.random_sc_diff = random_sc_diff
        self.fill_value = 0 if fill_value is None else (np.array(fill_value) * 255).reshape((1, 1, -1)).astype('uint8')
        self.margin = margin
        self.blind_crop = blind_crop  # don't try to validate cropping location, good for certain datasets
        self.interp_method = interp_method
        self.only_crop = only_crop
        self.return_bounds = return_bounds

    def most_ok_in_window(self, mask, sc=4):
        b = self.margin
        n, m = (np.array(self.shape) - b * 2) // sc
        mask_sc = cv2.resize(mask[b:-b, b:-b].astype(np.float32), None, fx=1/sc, fy=1/sc, interpolation=cv2.INTER_AREA)

        res = cv2.filter2D(mask_sc, ddepth=cv2.CV_32F, anchor=(0, 0),
                           kernel=np.ones((n, m), dtype='float32') * (1 / m / n),
                           borderType=cv2.BORDER_ISOLATED)
        res[res.shape[0] - n - 1:, :] = 0
        res[:, res.shape[1] - m - 1:] = 0
        res[res < np.max(res) * 0.5] = 0  # only possible to select windows that are at most half as bad as best window

        a = np.cumsum(res.flatten())
        if a[-1] > 0:
            rnd_idx = np.argmax(a > np.random.uniform(0, a[-1]))
            bst_idx = np.argmax(res.flatten())
        else:
            rnd_idx = bst_idx = 0

        bst_idxs = np.array(np.unravel_index(bst_idx, res.shape)) * sc + b
        bst_idxs = np.maximum(np.minimum(bst_idxs, np.subtract(mask.shape, self.shape)), 0)

        rnd_idxs = np.array(np.unravel_index(rnd_idx, res.shape)) * sc + b
        rnd_idxs = np.maximum(np.minimum(rnd_idxs, np.subtract(mask.shape, self.shape)), 0)

        return bst_idxs, rnd_idxs

    def __call__(self, imgs, aflow, *meta, debug=False):
        img1, img2 = imgs
        n, m = self.shape
        ow1, oh1 = img1.size
        ow2, oh2 = img2.size

        if (m > img1.size[0] or n > img1.size[1]) and not self.only_crop:
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

        assert self.only_crop or np.all(np.array((*bst_idxs, *rnd_idxs)) >= 0), \
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
            c_aflow = aflow[j1:j1+n, i1:i1+m, :]

            # if too few valid correspondences, pick the central crop instead
            ratio_valid = 1 - np.mean(np.isnan(c_aflow[:, :, 0]))
            if ratio_valid > 0.05:
                break

        assert self.only_crop or i1 + m <= aflow.shape[1] and j1 + n <= aflow.shape[0], \
            'invalid crop origin (%d, %d) for window size (%d, %d) and image size (%d, %d), is_random: %s' % (
                i1, j1, m, n, aflow.shape[1], aflow.shape[0], is_random)

        if ratio_valid == 0:
            # if this becomes a real problem, use SafeDataset and SafeDataLoader from nonechucks, then return None here
            if 0:
                from navex.datasets.base import DataLoadingException
                raise DataLoadingException("no valid correspondences even for central crop")
            print("no valid correspondences even for central crop")
            i1e, j1e = min(i1+m, img1.size[0]), min(j1+n, img1.size[1])
            i2e, j2e = min(m, img2.size[0]), min(n, img2.size[1])
            c_img1 = img1.crop((i1, j1, i1e, j1e))
            c_img2 = img2.crop((0, 0, i2e, j2e))
            return (c_img1, c_img2), c_aflow, *meta, \
                *((((i1, j1, i1e, j1e, ow1, oh1), (0, 0, i2e, j2e, ow2, oh2)),) if self.return_bounds else tuple())

        c_img1 = img1.crop((i1, j1, min(i1+m, img1.size[0]), min(j1+n, img1.size[1])))
        c_mask = mask[j1:j1+n, i1:i1+m]
        m1, n1 = c_img1.size

        # determine current scale of cropped img1 relative to cropped img0 based on aflow
        xy1 = np.stack(np.meshgrid(range(m1), range(n1)), axis=2).reshape((-1, 2))
        ic1, jc1 = np.median(xy1[c_mask.flatten(), :], axis=0)
        sc1 = np.sqrt(np.median(np.sum((xy1[c_mask.flatten(), :] - np.array((ic1, jc1)))**2, axis=1)))
        ic2, jc2 = np.nanmedian(c_aflow, axis=(0, 1))
        sc2 = np.sqrt(np.nanmedian(np.sum((c_aflow - np.array((ic2, jc2)))**2, axis=2)))
        curr_sc = np.clip(sc2 / (sc1 + 1e-8), 1/5, 5)  # limit to reasonable original scale range between the image pair

        # determine target scale based on current scale, self.max_sc_diff, and self.random_sc_diff
        lsc = abs(np.log10(self.max_sc_diff or 1e10))
        if self.only_crop:
            trg_sc = curr_sc
        elif self.random_sc_diff and is_random:
            # if first try fails, don't scale for second try
            trg_sc = 10**np.random.uniform(-lsc, lsc)
        else:
            min_sc, max_sc = 10 ** (-lsc), 10 ** lsc
            trg_sc = np.clip(curr_sc, min_sc, max_sc)

        cm, cn = math.ceil(m * curr_sc / trg_sc), math.ceil(n * curr_sc / trg_sc)
        if (cm > img2.size[0] or cn > img2.size[1]) and not self.only_crop:
            # padding is necessary
            img2, c_aflow = self._pad((cm, cn), img2, c_aflow, first=False)

        # scale aflow
        c_aflow = c_aflow * trg_sc/curr_sc
        trg_full_shape = math.ceil(img2.size[1] * trg_sc / curr_sc), math.ceil(img2.size[0] * trg_sc / curr_sc)

        if self.blind_crop:
            i2, j2 = (np.nanmean(c_aflow, axis=(0, 1)) - np.array([m/2, n/2]) + 0.5).astype(np.int)
            i2, j2 = np.clip(i2, 0, trg_full_shape[1] - m), np.clip(j2, 0, trg_full_shape[0] - n)
        else:
            # use cv2.filter2D and argmax for img1 also
            idxs = c_aflow.reshape((-1, 2))[np.logical_not(np.isnan(c_aflow[:, :, 0].flatten())), :].astype(np.int)
            idxs = idxs[np.logical_and(idxs[:, 0] < trg_full_shape[1], idxs[:, 1] < trg_full_shape[0]), :]
            c_ok = np.zeros(trg_full_shape, dtype='float32')
            c_ok[idxs[:, 1], idxs[:, 0]] = 1
            (j2, i2), _ = self.most_ok_in_window(c_ok)

        # crop and resize image 2
        i2s, i2e, j2s, j2e = (np.array((i2, i2+m, j2, j2+n))*curr_sc/trg_sc + 0.5).astype(np.int)
        i2s, j2s = max(0, i2s), max(0, j2s)
        i2e, j2e = min(img2.size[0], i2e), min(img2.size[1], j2e)

        try:
            c_img2 = img2.crop((i2s, j2s, i2e, j2e))
            if not self.only_crop:
                c_img2 = c_img2.resize((m, n), self.interp_method)
            m2, n2 = c_img2.size
        except PIL.Image.DecompressionBombError as e:
            from navex.datasets.base import DataLoadingException
            raise DataLoadingException((
                "invalid crop params? (i2s, j2s, i2e, j2e): %s, img1.size: %s, "
                "sc1: %s, sc2: %s, curr_sc: %s, trg_sc: %s"
                ) % ((i2s, j2s, i2e, j2e), img2.size, sc1, sc2, curr_sc, trg_sc)) from e

        # massage aflow
        c_aflow = (c_aflow - np.array((i2, j2), dtype=c_aflow.dtype)).reshape((-1, 2))
        c_aflow[np.any(c_aflow < 0, axis=1), :] = np.nan
        c_aflow[np.logical_or(c_aflow[:, 0] > m2 - 1, c_aflow[:, 1] > n2 - 1), :] = np.nan
        c_aflow = c_aflow.reshape((n1, m1, 2))

        if debug or 0:
            show_pair(c_img1, c_img2, c_aflow, pts=20)
            # min_i, min_j = np.nanmin(c_aflow, axis=(0, 1))
            # max_i, max_j = np.nanmax(c_aflow, axis=(0, 1))
            # assert min_i >= 0 and min_j >= 0, 'flow coord less than zero: i: %s, j: %s' % (min_i, min_j)
            # assert max_i < m and max_j < n, 'flow coord greater than cropped size: i: %s, j: %s' % (max_i, max_j)

        assert self.only_crop or tuple(c_img1.size) == tuple(np.flip(self.shape)), 'Image 1 is wrong size: %s' % (c_img1.size,)
        assert self.only_crop or tuple(c_img2.size) == tuple(np.flip(self.shape)), 'Image 2 is wrong size: %s' % (c_img2.size,)
        assert self.only_crop or tuple(c_aflow.shape[:2]) == tuple(self.shape), 'Absolute flow is wrong shape: %s' % (c_aflow.shape,)

        return (c_img1, c_img2), c_aflow, *meta, \
            *((((i1, j1, i1+m, j1+n, ow1, oh1), (i2s, j2s, i2e, j2e, ow2, oh2)),) if self.return_bounds else tuple())

    def _pad(self, min_size, img, aflow, first):
        w, h = img.size
        nw, nh = max(min_size[0], w), max(min_size[1], h)
        pl, pt = (nw - w) // 2, (nh - h) // 2
        pr, pb = nw - w - pl, nh - h - pt

        p_img = np.pad(np.atleast_3d(np.array(img)), ((pt, pb), (pl, pr), (0, 0)), mode='edge')
        p_img = PIL.Image.fromarray(p_img.squeeze())

        if first:
            p_aflow = np.ones((nh, nw, 2), dtype=np.float32) * np.nan
            p_aflow[pt:pt + h, pl:pl + w, :] = aflow
        else:
            p_aflow = aflow + np.array([pl, pt], dtype=np.float32)

        return p_img, p_aflow


class PairCenterCrop(PairRandomCrop):
    def __init__(self, shape, margin, max_sc_diff=None, fill_value=None, blind_crop=False,
                 only_crop=False, return_bounds=False):
        super(PairCenterCrop, self).__init__(shape, random=0, margin=margin, max_sc_diff=max_sc_diff,
                                             random_sc_diff=False, blind_crop=blind_crop, fill_value=fill_value,
                                             only_crop=only_crop, return_bounds=return_bounds)

    def __call__(self, imgs, aflow, *meta, **kwargs):
        return super(PairCenterCrop, self).__call__(imgs, aflow, *meta, **kwargs)


class RandomHomography:
    def __init__(self, max_tr, max_rot, max_shear, max_proj, min_size, one_tranf_only=False, fill_value=np.nan,
                 crop_valid=True, image_only=False, rnd_coef=2):
        self.max_tr = max_tr
        self.max_rot = max_rot
        self.max_shear = max_shear
        self.max_proj = max_proj
        self.min_size = min_size
        self.one_tranf_only = one_tranf_only
        self.fill_value = fill_value
        self.crop_valid = crop_valid
        self.image_only = image_only
        self.rnd_coef = rnd_coef

        self.transforms = (['tx', 'ty'] if self.max_tr > 0 else []) \
                        + (['r'] if self.max_rot > 0 else []) \
                        + (['sx', 'sy'] if self.max_shear > 0 else []) \
                        + (['p1', 'p2'] if self.max_proj > 0 else [])

    def rand(self, min_v, max_v, zero=0):
        if self.rnd_coef == 1:
            return random.uniform(min_v, max_v)

        x = random.random() ** (1/self.rnd_coef)   # get values that are closer to extremes
        a = (min_v - zero) if random.random() > 0.5 else (max_v - zero)
        return a*x

    def random_H(self, w, h):
        trs = [random.choice(self.transforms)] if self.one_tranf_only else self.transforms

        tr_x = self.rand(-self.max_tr, self.max_tr) * w if 'tx' in trs else 0
        tr_y = self.rand(-self.max_tr, self.max_tr) * h if 'ty' in trs else 0
        rot = self.rand(-self.max_rot, self.max_rot) if 'r' in trs else 0
        He = np.array([[math.cos(rot), -math.sin(rot), tr_x],
                       [math.sin(rot), math.cos(rot),  tr_y],
                       [0,             0,              1]], dtype=np.float32)

        sh_x = self.rand(-self.max_shear, self.max_shear) if 'sx' in trs else 0
        sh_y = self.rand(-self.max_shear, self.max_shear) if 'sy' in trs else 0
        Ha = np.array([[1, sh_x, 0],
                       [sh_y, 1, 0],
                       [0,    0, 1]], dtype=np.float32)

        p1 = self.rand(1/(1+self.max_proj) - 1, self.max_proj) / w if 'p1' in trs else 0
        p2 = self.rand(1/(1+self.max_proj) - 1, self.max_proj) / h if 'p2' in trs else 0
        Hp = np.array([[1,  0,  0],
                       [0,  1,  0],
                       [p1, p2, 1]], dtype=np.float32)

        H = He.dot(Ha).dot(Hp)
        return H

    def __call__(self, img, aflow=None):
        # from https://stackoverflow.com/questions/16682965/how-to-generaterandomtransform-with-opencv
        w, h = img.size
        if aflow is None:
            aflow = unit_aflow(w, h)
        aflow_shape = aflow.shape
        uh_aflow = np.concatenate((aflow, np.ones((*aflow_shape[:2], 1), dtype=np.float32)), axis=2)

        ok = False
        bad_h = 0
        for i in range(10):
            H = self.random_H(w, h)
            w_aflow = uh_aflow.reshape((-1, 3)).dot(H.T)
            if np.any(np.isclose(w_aflow[:, 2:], 0)):
                bad_h += 1
                continue

            w_aflow = (w_aflow[:, :2] / w_aflow[:, 2:]).reshape(aflow_shape)
            corners = w_aflow[[0, 0, -1, -1], [0, -1, 0, -1], :]
            (x0, y0), (x1, y1) = np.min(corners, axis=0), np.max(corners, axis=0)
            nw, nh = math.ceil(x1 - x0), math.ceil(y1 - y0)

            if nw >= self.min_size and nh >= self.min_size:
                ok = True
                break

        assert ok, ('Failed to generate valid homography, '
                    'resulting new size %s is less than the required %d, source size was %s (or bad H? %d/5)') % (
                        (nw, nh), self.min_size, (w, h), bad_h)

        w_aflow -= np.array([x0, y0]).reshape((1, 1, 2))

        if 1:
            tH = np.array([[1, 0, -x0],
                           [0, 1, -y0],
                           [0, 0, 1]]).dot(H)

            if self.crop_valid:
                wp_args = dict(borderMode=cv2.BORDER_CONSTANT, borderValue=self.fill_value)
            else:
                wp_args = dict(borderMode=cv2.BORDER_REPLICATE)

            img_arr = np.array(img, dtype=np.float32) / 255
            img_arr = cv2.warpPerspective(img_arr, tH, (nw, nh), flags=cv2.INTER_LINEAR, **wp_args)
            img_arr = 255 * img_arr + 0.5
        else:
            # old way, using scipy, which is slower
            uh_grid = np.concatenate((unit_aflow(nw, nh) + np.array([[[x0, y0]]]),
                                      np.ones((nh, nw, 1), dtype=np.float32)), axis=2)
            grid = uh_grid.reshape((-1, 3)).dot(np.linalg.inv(H.T))
            grid = (grid[:, :2] / grid[:, 2:]).reshape((nh, nw, 2))

            ifun = interp.RegularGridInterpolator((np.arange(h), np.arange(w)), np.array(img), bounds_error=False,
                                                  fill_value=np.array(self.fill_value)*255)
            img_arr = ifun(np.flip(grid, axis=2))

        if np.any(np.isnan(self.fill_value)) and self.crop_valid:
            mask = np.logical_not(np.isnan(np.atleast_3d(img_arr)[:, :, 0]))
            x0, y0, x1, y1 = tools.max_rect_bounded_by_quad_mask(mask)
            img_arr = img_arr[y0:y1, x0:x1]
            w_aflow = (w_aflow - np.array((x0, y0), dtype=w_aflow.dtype)).reshape((-1, 2))
            w_aflow[np.any(w_aflow < 0, axis=1), :] = np.nan
            w_aflow[np.logical_or(w_aflow[:, 0] > img_arr.shape[1] - 1, w_aflow[:, 1] > img_arr.shape[0] - 1), :] = np.nan
            w_aflow = w_aflow.reshape(aflow_shape)

        w_img = PIL.Image.fromarray(img_arr.astype(np.uint8))
        return w_img if self.image_only else (w_img, w_aflow)


class RandomHomography2(RandomHomography):
    def __call__(self, imgs, aflow, *meta):
        img1, img2 = imgs
        img2, aflow = super(RandomHomography2, self).__call__(img2, aflow=aflow)
        return (img1, img2), aflow, *meta


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
        return img * gain

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'min_gain={0}'.format(self.min_gain)
        format_string += ', max_gain={0})'.format(self.max_gain)
        return format_string


class Clamp:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, img):
        return torch.clamp(img, self.min, self.max)

    def __repr__(self):
        return self.__class__.__name__ + '(min={0}, max={1})'.format(self.min, self.max)


class GaussianNoise:
    def __init__(self, sd):
        self.sd = sd

    def __call__(self, img):
        return img + self.sd * torch.randn_like(img)

    def __repr__(self):
        return self.__class__.__name__ + '(sd={0})'.format(self.sd)


class UniformNoise:
    def __init__(self, ampl):
        self.ampl = ampl

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            return img + self.ampl * torch.rand_like(img) - 0.5*self.ampl
        else:
            img = np.array(img, dtype=np.float32)
            img = np.clip(img + np.random.uniform(0.5 - self.ampl / 2, 0.5 + self.ampl / 2, img.shape), 0, 255)
            return PIL.Image.fromarray(img.astype(np.uint8))

    def __repr__(self):
        return self.__class__.__name__ + '(ampl={0})'.format(self.ampl)


class RandomDarkNoise:
    # simulates dark current image sensor noise at different exposure levels

    def __init__(self, min_level, max_level, gain=0.008, pow=3):
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

        # dark current
        mean = random.uniform(self.min_level**(1/self.pow), self.max_level**(1/self.pow)) ** self.pow

        # dark shot noise (i.e. photon noise of dark current)
        sd = math.sqrt(self.gain * mean)

        return img + mean + sd * torch.randn_like(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'min_level={0}'.format(self.min_level)
        format_string += ', max_level={0}'.format(self.max_level)
        format_string += ', pow={0})'.format(self.pow)
        return format_string

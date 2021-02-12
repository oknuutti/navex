import itertools
import math
import os
import argparse
from collections import OrderedDict

from osgeo import gdal  # uses separate data_io conda env!

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator

from ..tools import unit_aflow, save_aflow, load_aflow, show_pair


def gen_aflow():
    parser = argparse.ArgumentParser('Get image pairs from an index file, '
                                     'based on *.xyz.exr and *.s.exr files, generate *.aflow.png file')
    parser.add_argument('--index', help="index file")
    parser.add_argument('--dst', help="output folder for the aflow files")
    args = parser.parse_args()

    src_path = os.path.dirname(args.index)
    os.makedirs(args.dst, exist_ok=True)

    groups = OrderedDict()
    with open(args.index, 'r') as fh:
        for line in fh:
            cells = line.split(' ')
            if len(cells) < 5:
                pass
            else:
                iid, gid, eid, fname = map(lambda x: x.strip(), cells[:4])
                if gid not in groups:
                    groups[gid] = {}
                groups[gid][eid] = (int(iid), fname)

    for gid, group in groups.items():
        for eid0, eid1 in itertools.combinations(group.keys(), 2):
            iid0, fname0 = group[eid0]
            iid1, fname1 = group[eid1]
            xyzs0 = load_xyzs(os.path.join(src_path, fname0))
            xyzs1 = load_xyzs(os.path.join(src_path, fname1))
            aflow = calc_aflow(xyzs0, xyzs1)
            save_aflow(os.path.join(args.dst, '%d_%d.aflow.png' % (iid0, iid1)), aflow)


def load_xyzs(fname, skip_s=False):
    if fname[-4:].lower() in ('.png', '.jpg'):
        fname = fname[:-4]

    xyz = cv2.imread(fname + '.xyz.exr', cv2.IMREAD_UNCHANGED)
    if skip_s:
        return xyz

    s = cv2.imread(fname + '.s.exr', cv2.IMREAD_UNCHANGED)
    return np.concatenate((xyz, np.atleast_3d(s)), axis=2)


def calc_aflow(xyzs0, xyzs1):
    h0, w0 = xyzs0.shape[:2]
    h1, w1 = xyzs1.shape[:2]

    img_xy1 = unit_aflow(h1, w1)
    img_xy1 = (img_xy1[:, :, 0] + 1j * img_xy1[:, :, 1]).flatten()

    # prepare x, doesnt support nans
    x = xyzs1[:, :, :3].reshape((-1, 3))
    I = np.logical_not(np.any(np.isnan(x), axis=1))
    len_sc = (np.nanmean(xyzs0[:, :, -1]) + np.nanmean(xyzs1[:, :, -1])) / 2

    interp = NearestKernelNDInterpolator(x[I, :], img_xy1[I], k_nearest=8, kernel_sc=len_sc/2, max_distance=len_sc * 3)
    aflow = interp(xyzs0[:, :, :3].reshape((-1, 3))).reshape((h0, w0))
    aflow = np.stack((np.real(aflow[:, :]), np.imag(aflow[:, :])), axis=2)

    return aflow


def create_image_pairs():
    parser = argparse.ArgumentParser('Group images based on their *.xyz.exr files')
    parser.add_argument('--root', help="image data root folder")
    parser.add_argument('--index', help="index file to use, or if missing, create (in root)")
    parser.add_argument('--pairs', help="pairing file to create in root")
    parser.add_argument('--src', default='', help="subfolder with the *.xyz.exr files")
    parser.add_argument('--aflow', help="subfolder where the aflow files are generated")
    parser.add_argument('--min-angle', type=float, default=5,
                        help="min angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--max-angle', type=float, default=20,
                        help="max angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--min-matches', type=int, default=10000,
                        help="min pixel matches in order to approve generated pair")
    parser.add_argument('--show-only', action='store_true', help="just show image pairs generated previously")
    args = parser.parse_args()

    aflow_path = os.path.join(args.root, args.aflow)
    index_path = os.path.join(args.root, args.index)
    pairs_path = os.path.join(args.root, args.pairs)
    src_path = os.path.join(args.root, args.src)

    os.makedirs(aflow_path, exist_ok=True)

    #  1) Go through all exr and find the median x, y, z; normalize to unity.
    #  2) Construct a tree from the unit vectors, find all pairs with max distance D.
    #  3) Try to construct aflow, reject if too few pixels match.
    #
    #  A better algorithm would be to create N clusters from pixel xyz and use the centroids in step 2)

    if not os.path.exists(index_path):
        print('building the index file...')

        # TODO: find and add here the direction of light in camera frame coords
        with open(index_path, 'w') as fh:
            fh.write('image_id image_file ux uy uz\n')

        files = [fname for fname in os.listdir(src_path) if fname[-8:].lower() == '.xyz.exr']
        for i, fname in enumerate(tqdm(sorted(files))):
            xyz = cv2.imread(os.path.join(src_path, fname), cv2.IMREAD_UNCHANGED)
            u = np.nanmedian(xyz.reshape((-1, 3)), axis=0)
            u /= np.linalg.norm(u)
            with open(index_path, 'a') as fh:
                fh.write(str(i) + ' ' + fname[:-8] + '.png ' + (' '.join('%f' % a for a in u)) + '\n')

    # read index file
    ids, files, unit_vectors = [], [], []
    with open(index_path, 'r') as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            id, fname, ux, uy, uz = map(lambda x: x.strip(), line.split(' '))
            ids.append(int(id))
            files.append(fname)
            unit_vectors.append([float(a) for a in (ux, uy, uz)])

    if args.show_only:
        show_all_pairs(aflow_path, args.root, dict(zip(ids, files)))
        return

    from scipy.spatial.ckdtree import cKDTree
    unit_vectors = np.array(unit_vectors)
    tree = cKDTree(unit_vectors)
    max_dist = 2 * math.sin(math.radians(args.max_angle) / 2)
    pairs = tree.query_pairs(max_dist, eps=0.05)

    print('building the pair file...')

    # TODO: make incremental, i.e. read existing pair file, skip in for loop if pair covered already

    with open(pairs_path, 'w') as fh:
        fh.write('image_id_0 image_id_1 angle match_ratio\n')

    added, pbar = 0, tqdm(pairs)
    for tot, (i, j) in enumerate(pbar):
        angle = math.degrees(2 * math.asin(np.linalg.norm(unit_vectors[i] - unit_vectors[j]) / 2))

        if angle >= args.min_angle:
            xyzs0 = load_xyzs(os.path.join(src_path, files[i]))
            xyzs1 = load_xyzs(os.path.join(src_path, files[j]))
            max_matches = min(np.sum(np.logical_not(np.isnan(xyzs0[:, :, 0]))),
                              np.sum(np.logical_not(np.isnan(xyzs1[:, :, 0]))))

            if max_matches >= args.min_matches:
                aflow = calc_aflow(xyzs0, xyzs1)
                matches = np.sum(np.logical_not(np.isnan(aflow[:, :, 0])))

                if matches >= args.min_matches:
                    added += 1
                    save_aflow(os.path.join(aflow_path, '%d_%d.aflow.png' % (ids[i], ids[j])), aflow)
                    with open(pairs_path, 'a') as fh:
                        fh.write('%d %d %.2f %.4f\n' % (ids[i], ids[j], angle, matches / max_matches))

        pbar.set_postfix({'ratio': added/(tot + 1)})


def show_all_pairs(aflow_path, img_path, image_db):
    for fname in os.listdir(aflow_path):
        if fname[-10:] == '.aflow.png':
            aflow = load_aflow(os.path.join(aflow_path, fname))
            id0, id1 = map(int, fname[:-10].split('_'))
            img0 = cv2.imread(os.path.join(img_path, image_db[id0]), cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(os.path.join(img_path, image_db[id1]), cv2.IMREAD_UNCHANGED)
            show_pair(img0, img1, aflow)


def read_raw_img(path, bands, gdtype=gdal.GDT_Float32, ndtype=np.float32):
    handle = gdal.Open(path, gdal.GA_ReadOnly)
    w, h, n = handle.RasterXSize, handle.RasterYSize, handle.RasterCount

    if max(bands) > n:
        raise Exception('file does not have enough bands: %d, needs: %s' % (n, bands))

    band_data = []
    for b in bands:
        bh = handle.GetRasterBand(b)
        raw = bh.ReadRaster(xoff=0, yoff=0, xsize=w, ysize=h, buf_xsize=w, buf_ysize=h, buf_type=gdtype)
        data = np.frombuffer(raw, dtype=ndtype).reshape((h, w))  # TODO: verify that order is h, w; not w, h
        band_data.append(data)
    data = np.stack(band_data, axis=2)

    # scale image to 8-bits
    img = (data[:, :, 0] * (255 / (2**12 - 1)) + 0.5).astype(np.uint8)

    return img, data[:, :, 1:]


def write_data(path, img, data):
    cv2.imwrite(path + '.png', img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    cv2.imwrite(path + '.xyz.exr', data[:, :, :3], (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
    cv2.imwrite(path + '.s.exr', data[:, :, 3:], (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))


class NearestKernelNDInterpolator(NearestNDInterpolator):
    def __init__(self, x, y, k_nearest=None, kernel='gaussian', kernel_sc=None,
                 kernel_eps=1e-12, query_eps=0.05, max_distance=None, **kwargs):
        """
        Parameters
        ----------
        kernel : one of the following functions of distance that give weight to neighbours:
            'linear': (kernel_sc/(r + kernel_eps))
            'quadratic': (kernel_sc/(r + kernel_eps))**2
            'cubic': (kernel_sc/(r + kernel_eps))**3
            'gaussian': exp(-(r/kernel_sc)**2)
        k_nearest : uses k_nearest neighbours for interpolation
        """
        choices = ('linear', 'quadratic', 'cubic', 'gaussian')
        assert kernel in choices, 'kernel must be one of %s' % (choices,)
        self._tree_options = kwargs.get('tree_options', {})

        assert len(y.shape), 'only one dimensional `y` supported'
        assert not np.any(np.isnan(x)), 'does not support nan values in `x`'

        super(NearestKernelNDInterpolator, self).__init__(x, y, **kwargs)
        if max_distance is None:
            if kernel_sc is None:
                d, _ = self.tree.query(self.points, k=k_nearest)
                kernel_sc = np.mean(d) * k_nearest / (k_nearest - 1)
            max_distance = kernel_sc * 3

        assert kernel_sc is not None, 'kernel_sc need to be set'
        self.kernel = kernel
        self.kernel_sc = kernel_sc
        self.kernel_eps = kernel_eps
        self.k_nearest = k_nearest
        self.max_distance = max_distance
        self.query_eps = query_eps

    def _linear(self, r):
        if scipy.sparse.issparse(r):
            return self.kernel_sc / (r + self.kernel_eps)
        else:
            return self.kernel_sc / (r + self.kernel_eps)

    def _quadratic(self, r):
        if scipy.sparse.issparse(r):
            return np.power(self.kernel_sc / (r.data + self.kernel_eps), 2, out=r.data)
        else:
            return (self.kernel_sc / (r + self.kernel_eps)) ** 2

    def _cubic(self, r):
        if scipy.sparse.issparse(r):
            return self.kernel_sc / (r + self.kernel_eps).power(3)
        else:
            return (self.kernel_sc / (r + self.kernel_eps)) ** 3

    def _gaussian(self, r):
        if scipy.sparse.issparse(r):
            return np.exp((-r.data / self.kernel_sc) ** 2, out=r.data)
        else:
            return np.exp(-(r / self.kernel_sc) ** 2)

    def __call__(self, *args):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        xi : ndarray of float, shape (..., ndim)
            Points where to interpolate data at.

        """
        from scipy.interpolate.interpnd import _ndim_coords_from_arrays

        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)

        r, idxs = self.tree.query(xi, self.k_nearest, eps=self.query_eps,
                                  distance_upper_bound=self.max_distance or np.inf)

        w = getattr(self, '_' + self.kernel)(r).reshape((-1, self.k_nearest)) + self.kernel_eps
        w /= np.sum(w, axis=1).reshape((-1, 1))

        # if idxs[i, j] == len(values), then i:th point doesnt have j:th match
        yt = np.concatenate((self.values, [np.nan]))

        yi = np.sum(yt[idxs] * w, axis=1)
        return yi


if __name__ == '__main__':
    create_image_pairs()

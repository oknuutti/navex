import itertools
import math
import os
import argparse
from collections import OrderedDict, Counter

# uses separate data_io conda env!
# create using:
#   conda create -n data_io -c conda-forge "python=>3.8" opencv matplotlib gdal geos tqdm scipy
from osgeo import gdal

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.cluster.vq import kmeans
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator

from navex.datasets.tools import unit_aflow, save_aflow, load_aflow, show_pair, ImageDB, find_files


def create_image_pairs_script():
    parser = argparse.ArgumentParser('Group images based on their *.xyz.exr files')
    parser.add_argument('--root', help="image data root folder")
    parser.add_argument('--index', help="index file to use, or if missing, create (in root)")
    parser.add_argument('--pairs', help="pairing file to create in root")
    parser.add_argument('--src', default='', help="subfolder with the *.xyz.exr files")
    parser.add_argument('--aflow', help="subfolder where the aflow files are generated")
    parser.add_argument('--img-max', type=int, default=3, help="how many times same images can be repated in pairs")
    parser.add_argument('--hz-fov', type=float,
                        help="horizontal FoV in degs, can be used with *.d.exr file if *.s.exr file missing")
    parser.add_argument('--min-angle', type=float, default=0,
                        help="min angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--max-angle', type=float, default=0,
                        help="max angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--min-matches', type=int, default=10000,
                        help="min pixel matches in order to approve generated pair")
    parser.add_argument('--read-meta', action='store_true', help="(re-)populate centroid info")
    parser.add_argument('--show-only', action='store_true', help="just show image pairs generated previously")

    args = parser.parse_args()
    create_image_pairs(root=args.root, index=args.index, pairs=args.pairs, src=args.src, aflow=args.aflow,
                       img_max=args.img_max, hz_fov=args.hz_fov, min_angle=args.min_angle, max_angle=args.max_angle,
                       min_matches=args.min_matches, read_meta=args.read_meta, show_only=args.show_only)


def create_image_pairs(root, index, pairs, src, aflow, img_max, hz_fov, min_angle,
                       max_angle, min_matches, read_meta, show_only=False, start=0.0, end=1.0):

    aflow_path = os.path.join(root, aflow)
    index_path = os.path.join(root, index)
    pairs_path = os.path.join(root, pairs)
    src_path = os.path.join(root, src)

    os.makedirs(aflow_path, exist_ok=True)

    #  1) Go through all exr and create 4 clusters from pixel xyz, normalize the centroids to unity.
    #  2) Construct a tree from the unit vectors, find all pairs with max distance D.
    #  3) Try to construct aflow, reject if too few pixels match.

    is_new = False
    if not os.path.exists(index_path):
        print('building the index file...')
        index = ImageDB(index_path, truncate=True)
        files = find_files(src_path, ext='.xyz.exr', relative=True)
        files = [(i, file) for i, file in enumerate(files)]
        is_new = True
    else:
        index = ImageDB(index_path)
        if read_meta:
            files = [(id, file[:-4] + '.xyz.exr') for id, file in
                     index.get_all(('id', 'file'), cond='cx1 is null', start=start, end=end)]

    if is_new or read_meta:
        # TODO: find and add here the direction of light in camera frame coords

        values = []
        print('clustering *.xyz.exr file contents...')
        for i, fname in tqdm(files):
            try:
                xyz = cv2.imread(os.path.join(src_path, fname), cv2.IMREAD_UNCHANGED).reshape((-1, 3))
            except Exception as e:
                raise Exception('Failed to open file %s' % os.path.join(src_path, fname)) from e

            I = np.logical_not(np.any(np.isnan(xyz), axis=1))
            if np.sum(I) > 0:
                u, _ = kmeans(xyz[I, :], 4)
                u /= np.atleast_2d(np.linalg.norm(u, axis=1)).T
                vd = np.max(np.abs(u - np.mean(u, axis=0)))
            else:
                vd, u = np.nan, np.ones((12,)) * np.nan
            values.append((i, fname[:-8] + '.png', vd, *u.flatten()))
        if is_new:
            index.add(('id', 'file', 'vd', 'cx1', 'cy1', 'cz1', 'cx2', 'cy2', 'cz2',
                                           'cx3', 'cy3', 'cz3', 'cx4', 'cy4', 'cz4'), values)
        else:
            index.set(('id', 'file', 'vd', 'cx1', 'cy1', 'cz1', 'cx2', 'cy2', 'cz2',
                                           'cx3', 'cy3', 'cz3', 'cx4', 'cy4', 'cz4'), values)

    # read index file
    ids, files, unit_vectors, max_dists = [], [], [], []
    for id, fname, vd, cx1, cy1, cz1, cx2, cy2, cz2, cx3, cy3, cz3, cx4, cy4, cz4 \
            in index.get_all(('id', 'file', 'vd', 'cx1', 'cy1', 'cz1', 'cx2', 'cy2', 'cz2',
                              'cx3', 'cy3', 'cz3', 'cx4', 'cy4', 'cz4'), start=start, end=end):
        assert cx1 is not None, 'centroids not set for id=%d, file=%s' % (id, fname)
        if np.isnan(float(cx1)):
            continue
        ids.append(int(id))
        files.append(fname)
        max_dists.append(float(vd))
        unit_vectors.append([[float(a) for a in u] for u in ((cx1, cy1, cz1), (cx2, cy2, cz2),
                                                             (cx3, cy3, cz3), (cx4, cy4, cz4))])

    if show_only:
        show_all_pairs(aflow_path, root, dict(zip(ids, files)))
        return

    from scipy.spatial.ckdtree import cKDTree
    unit_vectors = np.array(unit_vectors)
    tree = cKDTree(unit_vectors.reshape((-1, 3)))
    if max_angle > 0:
        max_dist = 2 * math.sin(math.radians(max_angle) / 2)
    else:
        max_dist = np.median(max_dists)
    pairs = tree.query_pairs(max_dist, eps=0.05)

    print('building the pair file...')
    added_pairs = set()
    image_count = Counter()
    if not os.path.exists(pairs_path):
        with open(pairs_path, 'w') as fh:
            fh.write('image_id_0 image_id_1 angle match_ratio\n')
    else:
        with open(pairs_path, 'r') as fh:
            for k, line in enumerate(fh):
                if k == 0:
                    continue
                i, j, *_ = line.split(' ')
                added_pairs.update({(int(i), int(j)), (int(j), int(i))})
                image_count.update((int(i), int(j)))

    print('calculating aflow for qualifying pairs...')
    pbar = tqdm(pairs)
    add_count = 0
    for tot, (ii, jj) in enumerate(pbar):
        i, j = ii // 4, jj // 4
        a, b = ii % 4, jj % 4
        angle = math.degrees(2 * math.asin(np.linalg.norm(unit_vectors[i][a] - unit_vectors[j][b]) / 2))

        if angle >= min_angle and i != j and (i, j) not in added_pairs and (
                image_count[i] < img_max or image_count[j] < img_max):
            xyzs0 = load_xyzs(os.path.join(src_path, files[i]), hz_fov=hz_fov)
            xyzs1 = load_xyzs(os.path.join(src_path, files[j]), hz_fov=hz_fov)
            max_matches = min(np.sum(np.logical_not(np.isnan(xyzs0[:, :, 0]))),
                              np.sum(np.logical_not(np.isnan(xyzs1[:, :, 0]))))

            if max_matches >= min_matches:
                aflow = calc_aflow(xyzs0, xyzs1)
                matches = np.sum(np.logical_not(np.isnan(aflow[:, :, 0])))

                if matches >= min_matches:
                    add_count += 1
                    added_pairs.add((i, j))
                    added_pairs.add((j, i))
                    image_count.update((i, j))
                    save_aflow(os.path.join(aflow_path, '%d_%d.aflow.png' % (ids[i], ids[j])), aflow)
                    with open(pairs_path, 'a') as fh:
                        fh.write('%d %d %.2f %.4f\n' % (ids[i], ids[j], angle, matches / max_matches))

        pbar.set_postfix({'added':add_count, 'ratio': add_count/(tot + 1)})


def gen_aflow_script():
    parser = argparse.ArgumentParser('Get image pairs from an index file, '
                                     'based on *.xyz.exr and *.s.exr files, generate *.aflow.png file')
    parser.add_argument('--index', help="index file")
    parser.add_argument('--dst', help="output folder for the aflow files")
    parser.add_argument('--hz-fov', type=float,
                        help="horizontal FoV in degs, can be used with *.d.exr file if *.s.exr file missing")
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
            xyzs0 = load_xyzs(os.path.join(src_path, fname0), hz_fov=args.hz_fov)
            xyzs1 = load_xyzs(os.path.join(src_path, fname1), hz_fov=args.hz_fov)
            aflow = calc_aflow(xyzs0, xyzs1)
            save_aflow(os.path.join(args.dst, '%d_%d.aflow.png' % (iid0, iid1)), aflow)


def load_xyzs(fname, skip_s=False, hz_fov=None):
    if fname[-4:].lower() in ('.png', '.jpg'):
        fname = fname[:-4]

    xyz = cv2.imread(fname + '.xyz.exr', cv2.IMREAD_UNCHANGED)
    if skip_s:
        return xyz

    if hz_fov is not None and not os.path.exists(fname + '.s.exr'):
        d = cv2.imread(fname + '.d.exr', cv2.IMREAD_UNCHANGED)
        coef = 2 * math.sin(math.radians(hz_fov / d.shape[1]) / 2)
        s = d * coef
    else:
        s = cv2.imread(fname + '.s.exr', cv2.IMREAD_UNCHANGED)

    return np.concatenate((xyz, np.atleast_3d(s)), axis=2)


def calc_aflow(xyzs0, xyzs1):
    h0, w0 = xyzs0.shape[:2]
    h1, w1 = xyzs1.shape[:2]

    img_xy1 = unit_aflow(h1, w1)
    img_xy1 = (img_xy1[:, :, 0] + 1j * img_xy1[:, :, 1]).flatten()

    # prepare x, doesnt support nans
    x = xyzs1[:, :, :3].reshape((-1, 3))
    I = np.logical_not(np.isnan(x[:, 0]))
    len_sc = max(np.nanquantile(xyzs0[:, :, -1], 0.9), np.nanquantile(xyzs1[:, :, -1], 0.9))

    interp = NearestKernelNDInterpolator(x[I, :], img_xy1[I], k_nearest=8, kernel_sc=len_sc, max_distance=len_sc * 3)
    aflow = interp(xyzs0[:, :, :3].reshape((-1, 3))).reshape((h0, w0))
    aflow = np.stack((np.real(aflow[:, :]), np.imag(aflow[:, :])), axis=2)

    return aflow


def show_all_pairs(aflow_path, img_path, image_db):
    for fname in os.listdir(aflow_path):
        if fname[-10:] == '.aflow.png':
            aflow = load_aflow(os.path.join(aflow_path, fname))
            id0, id1 = map(int, fname[:-10].split('_'))
            img0 = cv2.imread(os.path.join(img_path, image_db[id0]), cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(os.path.join(img_path, image_db[id1]), cv2.IMREAD_UNCHANGED)
            show_pair(img0, img1, aflow, image_db[id0], image_db[id1])


def read_raw_img(path, bands, gdtype=gdal.GDT_Float32, ndtype=np.float32, gamma=1.0):
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

    # scale and reduce depth to 8-bits
    top_v = np.quantile(data[:, :, 0], 0.999)
    if gamma == 1:
        img = np.clip((0.95 * 255 / top_v) * data[:, :, 0] + 0.5, 0, 255).astype(np.uint8)
    else:
        img = (255 * np.clip((0.95 / top_v) * data[:, :, 0], 0, 1) ** (1 / gamma)).astype(np.uint8)

    return img, data[:, :, 1:]


def write_data(path, img, data, xyzd=False):
    cv2.imwrite(path + '.png', img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    cv2.imwrite(path + '.xyz.exr', data[:, :, :3], (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
    if data.shape[2] > 3:
        cv2.imwrite(path + ('.d.exr' if xyzd else '.s.exr'), data[:, :, 3:],
                    (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))


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
    create_image_pairs_script()

import itertools
import math
import os
import argparse
import logging
import datetime
import re
import time
from collections import OrderedDict, Counter
import urllib
from typing import Callable
import json
import logging

from osgeo import gdal
import pvl

import numpy as np
import quaternion
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.cluster.vq import kmeans
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator

from navex.datasets.tools import unit_aflow, save_aflow, load_aflow, show_pair, ImageDB, find_files
from navex.experiments.parser import nested_filter


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
    index_path = index if isinstance(index, ImageDB) else os.path.join(root, index)
    pairs_path = os.path.join(root, pairs)
    src_path = os.path.join(root, src)

    os.makedirs(aflow_path, exist_ok=True)

    #  1) Go through all exr and create 4 clusters from pixel xyz, normalize the centroids to unity.
    #  2) Construct a tree from the unit vectors, find all pairs with max distance D.
    #  3) Try to construct aflow, reject if too few pixels match.

    is_new = False
    if isinstance(index_path, ImageDB) or os.path.exists(index_path):
        index = index_path if isinstance(index_path, ImageDB) else ImageDB(index_path)
    else:
        index = ImageDB(index_path, truncate=True)
        is_new = True

    if is_new:
        logging.info('building the index file...')
        files = find_files(src_path, ext='.xyz.exr', relative=True)
        files = [(i, file) for i, file in enumerate(files)]
    else:
        files = [(id, file[:-4] + '.xyz.exr') for id, file in
                 index.get_all(('id', 'file'), cond='cx1 is null', start=start, end=end)]

    if is_new or read_meta:
        logging.info('clustering *.xyz.exr file contents...')
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

            values = [(i, fname[:-8] + '.png', vd, *u.flatten())]
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

    logging.info('building the pair file...')
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
                i, j, angle, ratio = line.split(' ')
                added_pairs.update({(int(i), int(j)), (int(j), int(i))})
                if ratio != 'nan':
                    image_count.update((int(i), int(j)))

    logging.info('calculating aflow for qualifying pairs...')
    pbar = tqdm(pairs, mininterval=5)
    add_count = 0
    for tot, (ii, jj) in enumerate(pbar):
        i, j = ii // 4, jj // 4
        id_i, id_j = ids[i], ids[j]
        a, b = ii % 4, jj % 4
        angle = math.degrees(2 * math.asin(np.linalg.norm(unit_vectors[i][a] - unit_vectors[j][b]) / 2))

        if angle >= min_angle and id_i != id_j and (id_i, id_j) not in added_pairs and (
                image_count[id_i] < img_max or image_count[id_j] < img_max):
            xyzs0 = load_xyzs(os.path.join(src_path, files[i]), hz_fov=hz_fov)
            xyzs1 = load_xyzs(os.path.join(src_path, files[j]), hz_fov=hz_fov)
            max_matches = min(np.sum(np.logical_not(np.isnan(xyzs0[:, :, 0]))),
                              np.sum(np.logical_not(np.isnan(xyzs1[:, :, 0]))))

            ratio = np.nan
            if max_matches >= min_matches:
                aflow = calc_aflow(xyzs0, xyzs1)
                matches = np.sum(np.logical_not(np.isnan(aflow[:, :, 0])))

                if matches >= min_matches:
                    add_count += 1
                    image_count.update((id_i, id_j))
                    ratio = matches / max_matches
                    save_aflow(os.path.join(aflow_path, '%d_%d.aflow.png' % (id_i, id_j)), aflow)

            added_pairs.add((id_i, id_j))
            added_pairs.add((id_j, id_i))
            with open(pairs_path, 'a') as fh:
                fh.write('%d %d %.2f %.4f\n' % (id_i, id_j, angle, ratio))

        pbar.set_postfix({'added': add_count, 'ratio': add_count/(tot + 1)}, refresh=False)


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
    img_xy1 = (img_xy1[:, :, 0] + 1j * img_xy1[:, :, 1]).flatten().astype(np.complex64)

    # prepare x, doesnt support nans
    x = xyzs1[:, :, :3].reshape((-1, 3)).astype(np.float32)
    I = np.logical_not(np.isnan(x[:, 0]))
    len_sc = max(np.nanquantile(xyzs0[:, :, -1], 0.9), np.nanquantile(xyzs1[:, :, -1], 0.9))

    interp = NearestKernelNDInterpolator(x[I, :], img_xy1[I], k_nearest=8, kernel_sc=len_sc, max_distance=len_sc * 3)
    aflow = interp(xyzs0[:, :, :3].reshape((-1, 3)).astype(np.float32)).reshape((h0, w0))
    aflow = np.stack((np.real(aflow[:, :]), np.imag(aflow[:, :])), axis=2).astype(np.float32)

    return aflow


def show_all_pairs(aflow_path, img_path, image_db):
    for fname in os.listdir(aflow_path):
        if fname[-10:] == '.aflow.png':
            aflow = load_aflow(os.path.join(aflow_path, fname))
            id0, id1 = map(int, fname[:-10].split('_'))
            img0 = cv2.imread(os.path.join(img_path, image_db[id0]), cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(os.path.join(img_path, image_db[id1]), cv2.IMREAD_UNCHANGED)
            show_pair(img0, img1, aflow, image_db[id0], image_db[id1])


def read_raw_img(path, bands, gdtype=gdal.GDT_Float32, ndtype=np.float32, gamma=1.0, disp_dir=None,
                 metadata_type='esa/jaxa', q_wxyz=True, crop=None):
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

    if crop is not None:
        data = data[crop[0]:crop[1], crop[2]:crop[3], :]

    # scale and reduce depth to 8-bits
    bot_v, top_v = np.quantile(data[:, :, 0], (0.0005, 0.9999))
    top_v = top_v * 1.2
    img = (data[:, :, 0] - bot_v) / (top_v - bot_v)
    if gamma != 1:
        img = np.clip(img, 0, 1) ** (1 / gamma)
    img = np.clip(255 * img + 0.5, 0, 255).astype(np.uint8)

    handle = None
    if isinstance(metadata_type, Callable):
        allmeta, metadata, m_disp_dir = metadata_type(path)
    elif not metadata_type:
        allmeta, metadata, m_disp_dir = {}, {}, None
    else:
        allmeta, metadata, m_disp_dir = parse_metadata(path, q_wxyz)

    ele = {
        'forward': 'img = np.clip(((raw_img - bg) / max_val) ** (1 / gamma) + 0.5, 0, 255).astype(np.uint8)',
        'backward': 'raw_img = max_val * img ** gamma + bg',
        'bg': bot_v,
        'max_val': top_v - bot_v,
        'gamma': gamma,
        'possibly_corrupted_lines': np.sum(np.mean(data[:, :, 0] == 0, axis=1) > 0.95),
    }
    allmeta['image_processing'] = ele
    metadata['image_processing'] = ele

    # arrange data so that display direction is (down, right)
    disp_dir = m_disp_dir or disp_dir
    assert disp_dir is not None, 'LINE_DISPLAY_DIRECTION and SAMPLE_DISPLAY_DIRECTION specification missing'

    if disp_dir[0].lower() == 'up':
        img = np.flipud(img)
        data = np.flipud(data)
    if disp_dir[1].lower() == 'left':
        img = np.fliplr(img)
        data = np.fliplr(data)

    def default(o):
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.isoformat()
        if hasattr(o, 'tolist'):
            return o.tolist()
        raise TypeError('Can\'t serialize type %s' % (o.__class__,))

    meta_str = json.dumps(allmeta, sort_keys=True, indent=4, default=default)
    return img, data[:, :, 1:], metadata, meta_str


def parse_metadata(path, q_wxyz=True):
    # return metadata in J2000 frame
    meta = pvl.load(path)

    sc_rot_dec = metadata_value(meta, ('DECLINATION',), unit='rad')
    sc_rot_ra = metadata_value(meta, ('RIGHT_ASCENSION',), unit='rad')
    sc_rot_cna = metadata_value(meta, ('CELESTIAL_NORTH_CLOCK_ANGLE',), unit='rad')
    sc_ori_q = ypr_to_q(sc_rot_dec, sc_rot_ra, sc_rot_cna)
    if sc_ori_q is None:
        sc_ori_q, icrf_sc_pos = metadata_coord_frame(meta, ('SC_COORDINATE_SYSTEM', 'CAMERA_COORDINATE_SYSTEM'),
                                                      unit='km', q_wxyz=q_wxyz)

    # Can't figure this out for Hayabusa1 metadata
    sc_sun_pos_v = metadata_value(meta, ('SC_SUN_POSITION_VECTOR',), unit='km')

    # Can't figure this out for Hayabusa1 metadata
    sc_trg_pos_v = metadata_value(meta, ('SC_TARGET_POSITION_VECTOR',), unit='km')

    # cant figure this out for Rosetta or Hayabusa1 metadata
    trg_ori_q = None

    disp_dir, ld, sd = None, None, None
    if 'LINE_DISPLAY_DIRECTION' in meta and 'SAMPLE_DISPLAY_DIRECTION' in meta:
        ld, sd = meta['LINE_DISPLAY_DIRECTION'], meta['SAMPLE_DISPLAY_DIRECTION']
    elif 'IMAGE' in meta and 'LINE_DISPLAY_DIRECTION' in meta['IMAGE'] \
            and 'SAMPLE_DISPLAY_DIRECTION' in meta['IMAGE']:
        ld, sd = meta['IMAGE']['LINE_DISPLAY_DIRECTION'], meta['IMAGE']['SAMPLE_DISPLAY_DIRECTION']
    if isinstance(ld, str) and ld.lower() in ('up', 'down') \
            and isinstance(sd, str) and sd.lower() in ('left', 'right'):
        disp_dir = ld, sd

    metadata = {'sc_ori': sc_ori_q, 'sc_sun_pos': sc_sun_pos_v, 'trg_ori': trg_ori_q, 'sc_trg_pos': sc_trg_pos_v}
    return meta, metadata, disp_dir


def metadata_coord_frame(meta, coord_frame_keys, unit, q_wxyz=True):
    for k in coord_frame_keys:
        if k not in meta:
            return None, None

    pos = np.zeros((3,))
    ori = quaternion.one
    for k in coord_frame_keys:
        n_pos = metadata_value(meta[k], ['ORIGIN_OFFSET_VECTOR'], unit=unit)
        n_ori = meta[k]['ORIGIN_ROTATION_QUATERNION']
        n_ori = np.quaternion(*(n_ori if q_wxyz else (n_ori[3], n_ori[0], n_ori[1], n_ori[2])))
        pos = pos + q_times_v(ori, n_pos)
        ori = ori * n_ori

    return ori, pos


def metadata_value(meta, possible_keys, unit=''):
    src_values = None
    for key in possible_keys:
        if key in meta:
            src_values = meta[key]
            break

    is_scalar = False
    dst_values = []
    if src_values is not None:
        if src_values.__class__ not in (list, tuple):   # pvl.collections.Quantity has list or tuple as parent!
            src_values = [src_values]
            is_scalar = True

        for value in src_values:
            src_unit = value.units.lower().strip() if isinstance(value, pvl.collections.Quantity) else ''
            src_unit = re.sub('degrees|degree', 'deg', src_unit)
            src_unit = re.sub('radians|radian', 'rad', src_unit)
            src_unit = re.sub('kilometers|kilometer', 'km', src_unit)
            src_unit = re.sub('meters|meter', 'm', src_unit)

            value = value.value if isinstance(value, pvl.collections.Quantity) else value
            if src_unit in ('deg', 'rad'):
                assert unit in ('deg', 'rad'), f"can't transform {src_unit} to {unit}"
                if src_unit == 'deg' and unit == 'rad':
                    value = math.radians(value)
                elif src_unit != unit:
                    value = math.degrees(value)
            elif src_unit in ('m', 'km'):
                assert unit in ('m', 'km'), f"can't transform {src_unit} to {unit}"
                if src_unit == 'm' and unit == 'km':
                    value = value * 0.001
                elif src_unit != unit:
                    value = value * 1000
            elif not (src_unit == '' and unit == ''):
                logging.error("src_unit '%s' and dst_unit '%s' not handled currently" % (src_unit, unit))
                return None

            dst_values.append(value)
    else:
        dst_values = None

    return dst_values[0] if is_scalar else dst_values


def ypr_to_q(dec, ra, cna):
    if dec is None or ra is None or cna is None:
        return None

    # intrinsic euler rotations z-y'-x'', first right ascencion, then declination, and last celestial north angle
    return (
            np.quaternion(math.cos(ra / 2), 0, 0, math.sin(ra / 2))
            * np.quaternion(math.cos(-dec / 2), 0, math.sin(-dec / 2), 0)
            * np.quaternion(math.cos(-cna / 2), math.sin(-cna / 2), 0, 0)
    )


def q_times_v(q, v):
    qv = np.quaternion(0, *v)
    qv2 = q * qv * q.conj()
    return np.array([qv2.x, qv2.y, qv2.z])


def safe_split(x, is_q):
    if x is None:
        return (None,) * (4 if is_q else 3)
    return (*x[:3],) if not is_q else (x.w, x.x, x.y, x.z)


def check_img(img, bg_q=0.04, fg_q=240, sat_lo_q=0.998, sat_hi_q=0.9999, fg_lim=50, sat_lim=5, min_side=256):
    if fg_q > 1:
        # fg_q is instead the diameter of a half-circle in px
        fg_q = 1 - 0.5 * np.pi * (fg_q/2)**2 / np.prod(img.shape[:2])

    bg, fg, sat_lo ,sat_hi = np.quantile(img, (bg_q, fg_q, sat_lo_q, sat_hi_q))
    return np.min(img.shape) >= min_side and fg - bg >= fg_lim and sat_hi - sat_lo > sat_lim


def write_data(path, img, data, metastr=None, xyzd=False):
    cv2.imwrite(path + '.png', img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    if data is not None and data.size > 0:
        cv2.imwrite(path + '.xyz.exr', data[:, :, :3], (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
        if data.shape[2] > 3:
            cv2.imwrite(path + ('.d.exr' if xyzd else '.s.exr'), data[:, :, 3:],
                        (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT))
    if metastr is not None:
        with open(path + '.lbl', 'w') as fh:
            fh.write(metastr)


def get_file(url, path, max_retries=6):
    sleep_time = 3
    ok = False
    last_err = None
    for i in range(max_retries):
        try:
            urllib.request.urlretrieve(url, path)
            ok = True
            break
        except (urllib.error.ContentTooShortError, urllib.error.URLError) as e:
            last_err = e
            time.sleep(sleep_time)

    if not ok:
        raise Exception('Error: %s' % last_err)


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


def keys_to_lower(d):
    return nested_filter(d, lambda x: True, lambda x: x, lambda x: x.lower())


class DisableLogger:
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)


if __name__ == '__main__':
    create_image_pairs_script()

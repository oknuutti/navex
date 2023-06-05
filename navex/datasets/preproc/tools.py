import itertools
import math
import os
import argparse
import datetime
import re
import time
from collections import OrderedDict, Counter
import urllib
from typing import Callable
import json
import logging

import numpy as np
import quaternion
import matplotlib.pyplot as plt
import cv2
from scipy.cluster.vq import kmeans
from tqdm import tqdm

try:
    import numba as nb
except ImportError:
    nb = None
nb = None

from navex.datasets import tools
from ..tools import unit_aflow, save_aflow, load_aflow, show_pair, ImageDB, find_files, ypr_to_q, \
    q_times_v, angle_between_v, valid_asteriod_area, tf_view_unit_v, preprocess_image, rotate_array, Camera, \
    from_opencv_v, from_opencv_q, save_xyz, save_mono, load_xyz, load_mono, estimate_pose_pnp, estimate_pose_icp, \
    NearestKernelNDInterpolator, find_files_recurse
from navex.experiments.parser import nested_filter


def create_image_pairs_script():
    parser = argparse.ArgumentParser('Group images based on their *.xyz files')
    parser.add_argument('--root', help="image data root folder")
    parser.add_argument('--index', help="index file to use, or if missing, create (in root)")
    parser.add_argument('--pairs', help="pairing file to create in root")
    parser.add_argument('--src', default='', help="subfolder with the *.xyz files")
    parser.add_argument('--aflow', help="subfolder where the aflow files are generated")
    parser.add_argument('--img-max', type=int, default=3, help="how many times same images can be repated in pairs")
    parser.add_argument('--hz-fov', type=float,
                        help="horizontal FoV in degs, can be used with *.d file if *.s file missing")
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
                       img_max=args.img_max, def_hz_fov=args.hz_fov, min_angle=args.min_angle, max_angle=args.max_angle,
                       min_matches=args.min_matches, read_meta=args.read_meta, show_only=args.show_only)


def convert_from_exr_to_png():
    parser = argparse.ArgumentParser('Convert *.xyz.exr, *.d.exr and *.s.exr files to own png-based format')
    parser.add_argument('--root', help="image data root folder")
    parser.add_argument('--xyz2d', action='store_true', help="use .xyz file to generate .d file")
    parser.add_argument('--rm', action='store_true', help="remove converted files, default: no")
    args = parser.parse_args()

    from .eros_msi import CAM as msi
    from .itokawa_amica import CAM as amica
    from .cg67p_rosetta import INSTR
    from .synth import CAM as vcam
    cam = {
        'synth': vcam,
        'osinac': INSTR['osinac']['cam'],
        'eros': msi,
        'itokawa': amica,
    }.get(os.path.basename(args.root), None)

    assert not args.xyz2d or cam is not None, \
        'if use xyz2d arg, need cam (couldnt find with key %s)' % (os.path.basename(args.root),)

    files = find_files_recurse(args.root, ext=re.compile(r"^.*?(\.xyz\.exr|\.d\.exr|\.s\.exr)$"))
    for file in tqdm(files):
        if file.endswith('.xyz.exr'):
            xyz = load_xyz(file)
            if args.xyz2d:
                d = _convert_xyz2d(xyz, cam)
                save_mono(file[:-8]+'.d', d)
            save_xyz(file[:-4], xyz)
        else:
            if args.xyz2d and file.endswith('.d.exr'):
                pass
            else:
                x = load_mono(file)
                save_mono(file[:-4], x)

        if args.rm:
            os.unlink(file)


def _convert_xyz2d(xyz, cam):
    sc_trg_pos, sc_trg_ori = estimate_pose_pnp(cam, xyz, ba=False)
    v = xyz + q_times_v(sc_trg_ori.conj(), sc_trg_pos)
    d = np.linalg.norm(v, axis=2)
    return d


def create_image_pairs(root, index, pairs, geom_src, aflow, img_max, def_hz_fov, min_angle,
                       max_angle, min_matches, read_meta, max_sc_diff=1.5, max_dist=None, show_only=False, start=0.0, end=1.0,
                       exclude_shadowed=True, across_subsets=False, depth_src=None, cluster_unit_vects=True,
                       max_cluster_diff_angle=None, aflow_match_coef=1.0, trust_georef=True, ignore_img_angle=True,
                       depth_is_along_zaxis=False):
    aflow_path = os.path.join(root, aflow)
    index_path = index if isinstance(index, ImageDB) else os.path.join(root, index)
    pairs_path = os.path.join(root, pairs)
    geom_src_fun = (lambda fname: os.path.join(root, geom_src, fname)) if isinstance(geom_src, str) else geom_src

    depth_src = depth_src or geom_src
    depth_src_fun = (lambda fname: os.path.join(root, depth_src, fname[:-4] + '.d')) if isinstance(depth_src, str) else depth_src

    os.makedirs(aflow_path, exist_ok=True)

    #  1) Go through all exr and create 4 clusters from pixel xyz, normalize the centroids to unity.
    #  2) Construct a tree from the unit vectors, find all pairs with max distance D.
    #  3) Filter out pairs that fail one of these conditions:
    #       - scale difference at most max_sc_diff
    #       - relative angle of view between min_angle and max_angle
    #       - both images already used img_max times
    #  4) Construct aflow for selected pairs, can still reject if too few pixels match (min_matches)

    is_new = False
    if isinstance(index_path, ImageDB) or os.path.exists(index_path):
        index = index_path if isinstance(index_path, ImageDB) else ImageDB(index_path)
    else:
        index = ImageDB(index_path, truncate=True)
        is_new = True

    if is_new:
        assert isinstance(geom_src, str), 'source path function not supported without existing index'
        logging.info('building the index file...')
        files = find_files(os.path.join(root, geom_src), ext='.xyz', relative=True)
        files = [(i, file[:-8] + '.png') for i, file in enumerate(files)]
    else:
        files = [(id, file) for id, file in
                 index.get_all(('id', 'file'), cond='vd is null', start=start, end=end)]

    if is_new or read_meta:
        for i, fname in tqdm(files, desc='Clustering *.xyz file contents'):
            try:
                xyz = load_xyzs(geom_src_fun(fname), skip_s=True, hide_shadow=exclude_shadowed).reshape((-1, 3))
            except Exception as e:
                raise Exception('Failed to open file %s' % geom_src_fun(fname)) from e

            I = np.logical_not(np.any(np.isnan(xyz), axis=1))
            vd, u = np.nan, np.ones((4, 3)) * np.nan
            if np.sum(I) >= min_matches:
                ut, _ = kmeans(xyz[I, :], 4)
                if cluster_unit_vects:
                    ut /= np.atleast_2d(np.linalg.norm(ut, axis=1)).T
                vd = np.max(np.abs(ut - np.mean(ut, axis=0)))
                u[:len(ut), :] = ut

            values = [(i, fname, vd, *u.flatten())]
            if is_new:
                index.add(('id', 'file', 'vd', 'cx1', 'cy1', 'cz1', 'cx2', 'cy2', 'cz2',
                                               'cx3', 'cy3', 'cz3', 'cx4', 'cy4', 'cz4'), values)
            else:
                index.set(('id', 'file', 'vd', 'cx1', 'cy1', 'cz1', 'cx2', 'cy2', 'cz2',
                                               'cx3', 'cy3', 'cz3', 'cx4', 'cy4', 'cz4'), values)

    # read index file
    ids, files, hz_fovs, angles, centroid_vects, max_dists, dists, coords, poses = [], [], [], [], [], [], [], [], []
    set_ids, cams = [], {}
    for id, set_id, fname, hz_fov, angle, vd, stx, sty, stz, sqw, sqx, sqy, sqz, tqw, tqx, tqy, tqz, \
        cx1, cy1, cz1, cx2, cy2, cz2, cx3, cy3, cz3, cx4, cy4, cz4 \
            in index.get_all(('id', 'set_id', 'file', 'hz_fov', 'img_angle', 'vd', 'sc_trg_x', 'sc_trg_y', 'sc_trg_z',
                              'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz', 'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz',
                              'cx1', 'cy1', 'cz1', 'cx2', 'cy2', 'cz2',
                              'cx3', 'cy3', 'cz3', 'cx4', 'cy4', 'cz4'), start=start, end=end):
        assert cx1 is not None, 'centroids not set for id=%d, file=%s' % (id, fname)
        if np.isnan(float(cx1)) or tqw is None:
            continue
        ids.append(int(id))
        set_ids.append(int(set_id or -1))
        files.append(fname)
        hz_fovs.append(hz_fov)
        angles.append(None if ignore_img_angle else angle)
        max_dists.append(float(vd))

        if set_id and set_id not in cams:
            cams[set_id] = cam_obj(index.get_subset(set_id))

        stv = np.array([stx, sty, stz])

        # cam orientation in target frame (not target orientation in cam frame)
        stq = ((np.quaternion(tqw, tqx, tqy, tqz) if tqw is not None else quaternion.one).conj()
               * (np.quaternion(sqw, sqx, sqy, sqz) if sqw is not None else quaternion.one))

        poses.append((stv, stq))
        dists.append(np.linalg.norm(stv))

        # cam axis in target frame, FIX(ed?): didn't work for Nokia data, coord frame different there?
        coords.append(tf_view_unit_v(stq))

        centroid_vects.append([[float(a) for a in u] for u in ((cx1, cy1, cz1), (cx2, cy2, cz2),
                                                               (cx3, cy3, cz3), (cx4, cy4, cz4))])

    if show_only:
        show_all_pairs(aflow_path, root, dict(zip(ids, files)))
        return

    from scipy.spatial.ckdtree import cKDTree
    centroid_vects = np.array(centroid_vects).reshape((-1, 3))
    I = np.logical_not(np.any(np.isnan(centroid_vects), axis=1))
    centroid_vects = centroid_vects[I, :]
    tree = cKDTree(centroid_vects)
    idxs = np.stack((np.repeat(np.atleast_2d(np.arange(0, len(ids))).T, 4, axis=1),
                     np.repeat(np.atleast_2d(np.arange(0, 4)), len(ids), axis=0)), axis=2).reshape((-1, 2))[I, :]

    if max_dist is None:
        if max_cluster_diff_angle and cluster_unit_vects:
            max_dist = 2 * math.sin(math.radians(max_cluster_diff_angle) / 2)
        else:
            max_dist = np.median(max_dists)

    pairs = tree.query_pairs(max_dist, eps=0.05)

    centroids_per_img = len(centroid_vects) / len(max_dists)
    mean_pair_cands = len(pairs) / len(centroid_vects)
    logging.info('Each image (n=%d) has %.2f centroids and %.3f pair candidates on average' % (
        len(max_dists), centroids_per_img, mean_pair_cands))

    added_pairs = set()
    image_count = Counter()
    if not os.path.exists(pairs_path):
        logging.info('Building the pair file...')
        with open(pairs_path, 'w') as fh:
            fh.write('image_id_0 image_id_1 sc_diff angle_diff match_ratio matches\n')
    else:
        logging.info('Reading the existing pair file...')
        with open(pairs_path, 'r') as fh:
            for k, line in enumerate(fh):
                if k == 0:
                    continue
                i, j, sc_diff, angle, ratio, matches = line.split(' ')
                added_pairs.update({(int(i), int(j)), (int(j), int(i))})
                if ratio != 'nan':
                    image_count.update((int(i), int(j)))

    pbar = tqdm(pairs, mininterval=5, desc='Calculating aflow for qualifying pairs')
    add_count = 0
    for tot, (ii, jj) in enumerate(pbar):
        (i, a), (j, b) = idxs[ii, :], idxs[jj, :]
        id_i, id_j = ids[i], ids[j]

        # calculate angle between camera axes in target frame
        angle = math.degrees(angle_between_v(coords[i], coords[j]))

        sc_diff = max(dists[i] / dists[j], dists[j] / dists[i])
        fi, fj = files[i], files[j]

        if min_angle <= angle <= max_angle and sc_diff < max_sc_diff and id_i != id_j \
                and (id_i, id_j) not in added_pairs and (image_count[id_i] < img_max or image_count[id_j] < img_max) \
                and (set_ids[i] != set_ids[j] if across_subsets else True):
            xyzs0 = load_xyzs(geom_src_fun(fi), d_file=depth_src_fun(fi), hide_shadow=exclude_shadowed,
                              hz_fov=hz_fovs[i] or def_hz_fov, xyzd=not trust_georef)
            xyzs1 = load_xyzs(geom_src_fun(fj), d_file=depth_src_fun(fj), hide_shadow=exclude_shadowed,
                              hz_fov=hz_fovs[j] or def_hz_fov, xyzd=not trust_georef)
            max_matches = min(np.sum(np.logical_not(np.isnan(xyzs0[:, :, 0]))),
                              np.sum(np.logical_not(np.isnan(xyzs1[:, :, 0]))))

            ratio, matches = np.nan, -1
            if max_matches >= min_matches:
                if trust_georef:
                    aflow = calc_aflow(xyzs0, xyzs1, uncertainty_coef=aflow_match_coef)
                else:
                    aflow = est_aflow(xyzs0, xyzs1, poses[i], poses[j], cams[set_ids[i]], cams[set_ids[j]],
                                      os.path.join(root, fi), os.path.join(root, fj), angles[i], angles[j],
                                      min_n=min_matches, depth_is_along_zaxis=depth_is_along_zaxis)

                if 0:
                    debug_aflow(aflow, os.path.join(root, fi), os.path.join(root, fj), angles[i], angles[j],
                                cams[set_ids[i]], cams[set_ids[j]], xyzs0, xyzs1)

                matches = np.sum(np.logical_not(np.isnan(aflow[:, :, 0])))

                if matches >= min_matches:
                    if angles[i] and angles[j]:
                        aflow = tools.rotate_aflow(aflow, xyzs1.shape[:2], angles[i], angles[j])

                    if 0:
                        debug_aflow(aflow, os.path.join(root, fi), os.path.join(root, fj), 0, 0,
                                    cams[set_ids[i]], cams[set_ids[j]], xyzs0, xyzs1)

                    add_count += 1
                    image_count.update((id_i, id_j))
                    ratio = matches / max_matches
                    save_aflow(os.path.join(aflow_path, '%d_%d.aflow.png' % (id_i, id_j)), aflow)

            added_pairs.add((id_i, id_j))
            added_pairs.add((id_j, id_i))
            with open(pairs_path, 'a') as fh:
                fh.write('%d %d %.3f %.3f %.3f %d\n' % (id_i, id_j, sc_diff, angle, ratio, matches))

        pbar.set_postfix({'added': add_count, 'ratio': add_count/(tot + 1)}, refresh=False)


def cam_obj(cam_params):
    width, height, fl_x, fl_y, pp_x, pp_y, *dist_coefs = cam_params
    cam_mx = np.array([[fl_x, 0, pp_x], [0, fl_y, pp_y], [0, 0, 1]])
    cam = Camera(cam_mx, [width, height], dist_coefs=dist_coefs)
    return cam


def gen_aflow_script():
    parser = argparse.ArgumentParser('Get image pairs from an index file, '
                                     'based on *.xyz and *.s files, generate *.aflow.png file')
    parser.add_argument('--index', help="index file")
    parser.add_argument('--dst', help="output folder for the aflow files")
    parser.add_argument('--hz-fov', type=float,
                        help="horizontal FoV in degs, can be used with *.d file if *.s file missing")
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


def load_xyzs(i_file, g_file=None, s_file=None, d_file=None, h_file=None, skip_s=False,
              hz_fov=None, hide_shadow=True, rotate_angle=None, xyzd=False):
    if i_file[-4:].lower() in ('.png', '.jpg'):
        g_file = g_file or (i_file[:-4] + '.xyz.exr')
    elif i_file[-8:] == '.xyz.exr':
        g_file = g_file or i_file
        i_file = i_file[-8:] + '.png'
    else:
        g_file = g_file or (i_file + '.xyz.exr')
        i_file = i_file + '.png'

    if g_file[-4:] == '.exr' and os.path.exists(g_file[:-4]):
        g_file = g_file[-4:]

    xyz = load_xyz(g_file)
    if xyz is None:
        raise FileNotFoundError("couldn't read file %s" % g_file)

    # hide parts that are in shadow
    if hide_shadow:
        h_file = h_file or (i_file[:-4] + '.sdw')
        if os.path.exists(h_file):
            mask = cv2.imread(h_file, cv2.IMREAD_UNCHANGED).astype(bool)
        else:
            img = cv2.imread(i_file, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError("couldn't read file %s" % (i_file,))
            mask = valid_asteriod_area(img, 50, remove_limb=False)
            mask = np.logical_not(mask)
            cv2.imwrite(h_file + '.png', mask.astype(np.uint8) * 255, (cv2.IMWRITE_PNG_COMPRESSION, 9))
            os.rename(h_file + '.png', h_file)

        shape = xyz.shape
        xyz = xyz.reshape((-1, 3))
        xyz[mask.flatten(), :] = np.nan
        xyz = xyz.reshape(shape)

    if skip_s:
        if rotate_angle:
            xyz = rotate_array(xyz, rotate_angle, new_size='full', border=cv2.BORDER_CONSTANT, border_val=np.nan)
        return xyz

    s_file = s_file or (i_file[:-4] + '.s.exr')
    if s_file[-4:] == '.exr' and os.path.exists(s_file[:-4]):
        s_file = s_file[-4:]

    d_file = d_file or (i_file[:-4] + '.d.exr')
    if d_file[-4:] == '.exr' and os.path.exists(d_file[:-4]):
        d_file = d_file[-4:]

    if hz_fov is not None and not os.path.exists(s_file) or xyzd:
        d = load_mono(d_file)
        if not xyzd:
            coef = 2 * math.sin(math.radians(hz_fov / d.shape[1]) / 2)
            s = d * coef
    else:
        s = load_mono(s_file)

    xyzs = np.concatenate((xyz, np.atleast_3d(d if xyzd else s)), axis=2)
    if rotate_angle:
        xyzs = rotate_array(xyzs, rotate_angle, new_size='full', border=cv2.BORDER_CONSTANT, border_val=np.nan)

    return xyzs


def calc_aflow(xyzs0, xyzs1, uncertainty_coef=1.0):
    h0, w0 = xyzs0.shape[:2]
    h1, w1 = xyzs1.shape[:2]

    img_xy1 = unit_aflow(w1, h1)
    img_xy1 = (img_xy1[:, :, 0] + 1j * img_xy1[:, :, 1]).flatten().astype(np.complex64)

    # prepare x, doesnt support nans
    x = xyzs1[:, :, :3].reshape((-1, 3)).astype(np.float32)
    I = np.logical_not(np.isnan(x[:, 0]))
    len_sc = max(np.nanquantile(xyzs0[:, :, -1], 0.9), np.nanquantile(xyzs1[:, :, -1], 0.9)) * uncertainty_coef

    interp = NearestKernelNDInterpolator(x[I, :], img_xy1[I], k_nearest=8, kernel_sc=len_sc, max_distance=len_sc * 3)
    aflow = interp(xyzs0[:, :, :3].reshape((-1, 3)).astype(np.float32)).reshape((h0, w0))
    aflow = np.stack((np.real(aflow[:, :]), np.imag(aflow[:, :])), axis=2).astype(np.float32)

    return aflow


def est_aflow(xyzd0, xyzd1, pose0, pose1, cam0, cam1, imgfile0, imgfile1, angle0, angle1, min_n, margin=10,
              depth_is_along_zaxis=False):
    # transform d0 so that close to d1
    ixy0 = unit_aflow(cam0.width, cam0.height)
    ixy1 = unit_aflow(cam1.width, cam1.height)
    rel_pose_ini = pose_diff(pose1, pose0)

    depth = 'z_off' if depth_is_along_zaxis else 'dist'
    c_xyz0 = cam0.backproject(ixy0[:, :, :2].reshape((-1, 2)), **{depth: xyzd0[:, :, 3].flatten()})
    c_xyz1 = cam1.backproject(ixy1[:, :, :2].reshape((-1, 2)), **{depth: xyzd1[:, :, 3].flatten()})
    c_xyz0_1 = tools.q_times_mx(rel_pose_ini[1], c_xyz0) + rel_pose_ini[0]

    I0 = np.logical_not(np.isnan(c_xyz0_1[:, 0]))
    I1 = np.logical_not(np.isnan(c_xyz1[:, 0]))
    cropped_P0 = c_xyz0_1[I0, :]
    cropped_P1 = c_xyz1[I1, :]

    lo0, hi0 = np.min(cropped_P0, axis=0), np.max(cropped_P0, axis=0)
    lo1, hi1 = np.min(cropped_P1, axis=0), np.max(cropped_P1, axis=0)
    I0[I0] = np.logical_and(np.all(cropped_P0 > lo1, axis=1), np.all(cropped_P0 < hi1, axis=1))
    I1[I1] = np.logical_and(np.all(cropped_P1 > lo0 - margin, axis=1), np.all(cropped_P1 < hi0 + margin, axis=1))
    cropped_P0 = c_xyz0_1[I0, :]
    cropped_P1 = c_xyz1[I1, :]

    if min(len(cropped_P0), len(cropped_P1)) < min_n:
        return np.ones(xyzd0.shape[:2] + (2,)) * np.nan

    if 1:
        # adjust scale
        med_z0 = np.median(cropped_P0[:, 2])
        med_z1 = np.median(cropped_P1[:, 2])
        cropped_P0 *= med_z1 / med_z0
        c_xyz0_1 *= med_z1 / med_z0

    # TODO: if either depth map is too flat (i.e. featureless) or err is large, use template matching on images instead
    if 0:
        # run ICP to get pose adjustment
        adj_rel_pos, adj_rel_ori, err = estimate_pose_icp(cropped_P0, cropped_P1, max_n1=200000, max_n2=100000)
    elif 0:
        # run template matching on images to get pose adjustment
        img0 = load_image(imgfile0, angle0, cam0)
        img1 = load_image(imgfile1, angle1, cam1)
        adj_rel_pos, adj_rel_ori, err = match_template(img0, img1, I0, cropped_P0, cam1)
    else:
        if 0:
            # seems to improve result marginally
            adj_rel_pos, adj_rel_ori, err = estimate_pose_icp(cropped_P0, cropped_P1, max_n1=200000, max_n2=100000)
            c_xyz0_1 = tools.q_times_mx(adj_rel_ori, c_xyz0_1) + adj_rel_pos
            cropped_P0 = tools.q_times_mx(adj_rel_ori, cropped_P0) + adj_rel_pos

        # run template matching on depth maps to get pose adjustment  (best results, very slow though)
        dm0 = c_xyz0_1[:, 2].reshape(xyzd1.shape[:2])
        adj_rel_pos, adj_rel_ori, err = match_template(dm0, xyzd1[:, :, 3], I0, cropped_P0, cam1, margin_px=120, skip=2,
                                           depthmap=True, use_edges=False)

        if 0:
            # seems to just make it worse
            c_xyz0_1 = tools.q_times_mx(adj_rel_ori, c_xyz0_1) + adj_rel_pos
            cropped_P0 = tools.q_times_mx(adj_rel_ori, cropped_P0) + adj_rel_pos
            adj_rel_pos, adj_rel_ori, err = estimate_pose_icp(cropped_P0, cropped_P1, max_n1=200000, max_n2=100000)

    if adj_rel_pos is None:
        return np.ones(xyzd0.shape[:2] + (2,)) * np.nan

    c_xyz0_1a = tools.q_times_mx(adj_rel_ori, c_xyz0_1) + adj_rel_pos

    # project to image-plane-1 to get aflow
    aflow = cam1.project(c_xyz0_1a)
    aflow[np.any(aflow < 0, axis=1), :] = np.nan
    aflow[np.logical_or(aflow[:, 0] > cam1.width - 1, aflow[:, 1] > cam1.height - 1), :] = np.nan
    aflow = aflow.reshape((cam0.height, cam0.width, 2))

    if 0:
        comp_aflow = cam1.project(c_xyz0_1)
        comp_aflow[np.any(comp_aflow < 0, axis=1), :] = np.nan
        comp_aflow[np.logical_or(comp_aflow[:, 0] > cam1.width - 1, comp_aflow[:, 1] > cam1.height - 1), :] = np.nan
        comp_aflow = comp_aflow.reshape((cam0.height, cam0.width, 2))
        debug_aflow(comp_aflow, imgfile0, imgfile1, angle0, angle1, cam0, cam1, xyzd0, xyzd1, comp_aflow=aflow,
                    plot_depth=False)

    return aflow


def load_image(imgfile, angle, cam):
    img = np.flip(cv2.imread(imgfile, cv2.IMREAD_COLOR), axis=2)

    if angle:
        img = rotate_array(img, -angle, new_size='full')
        cx, cy = (img.shape[1] - cam.width) / 2, (img.shape[0] - cam.height) / 2
        img = img[math.floor(cy):-math.ceil(cy), math.floor(cx):-math.ceil(cx)]

    return img


def project_image(img0, I0, P0_1, cam1):
    ixy = (cam1.project(P0_1) + 0.5).astype(int)
    I = np.logical_and.reduce((ixy[:, 0] >= 0, ixy[:, 0] < cam1.width,
                               ixy[:, 1] >= 0, ixy[:, 1] < cam1.height))
    I0[I0] = I
    ixy = ixy[I, :]
    img0 = img0.reshape((-1, img0.shape[2] if len(img0.shape) > 2 else 1))
    px_vals = img0[I0, :]

    # draw image
    img0_1 = np.ones((cam1.height, cam1.width, px_vals.shape[1]), dtype=np.float32) * np.nan
    img0_1[ixy[:, 1], ixy[:, 0], :] = px_vals

    return img0_1.squeeze()


def debug_aflow(aflow, imgfile0, imgfile1, angle0, angle1, cam0, cam1, xyzd0, xyzd1, comp_aflow=None, plot_depth=False):
    if plot_depth:
        img0 = xyzd0[:, :, 3]
        img1 = xyzd1[:, :, 3]
        min_val = max(np.nanmin(img0), np.nanmin(img1))
        max_val = min(np.nanmax(img0), np.nanmax(img1))
        img0 = np.clip(img0, min_val, max_val)
        img1 = np.clip(img1, min_val, max_val)
    else:
        img0 = load_image(imgfile0, angle0, cam0)
        img1 = load_image(imgfile1, angle1, cam1)

    fig, axs = plt.subplots(2, 2)
    show_pair(img0, img1, aflow, pts=20, axs=axs.flatten(), show=False)
    if comp_aflow is not None:
        show_pair(img0, img1, comp_aflow, pts=20, axs=axs.flatten()[2:], show=False)
    elif 1:
        axs[1][0].scatter(xyzd0[::2, ::2, 0].flatten(), xyzd0[::2, ::2, 1].flatten(), c=xyzd0[::2, ::2, 2].flatten(),
                          s=20, marker='o', vmin=-15., vmax=40.)
        axs[1][0].set_aspect('equal')
        axs[1][1].scatter(xyzd1[::2, ::2, 0].flatten(), xyzd1[::2, ::2, 1].flatten(), c=xyzd1[::2, ::2, 2].flatten(),
                          s=20, marker='o', vmin=-15., vmax=40.)
        axs[1][1].set_aspect('equal')
    else:
        axs[1][0].imshow(xyzd0[:, :, 3])
        axs[1][1].imshow(xyzd1[:, :, 3])
    plt.tight_layout()
    plt.show()


def pose_diff(pose1, pose0):
    """ returns: pose1 - pose0 """

    # -pose0
    npose0 = [tools.q_times_v(pose0[1].conj(), -pose0[0]), pose0[1].conj()]

    # -pose0 + pose1
    return [tools.q_times_v(pose1[1], npose0[0]) + pose1[0], (pose1[1] * npose0[1]).normalized()]


def match_template(img0, img1, I0, P0_1, cam1, margin_px=60, skip=1, depthmap=False, use_edges=False):
    def edges(img):
        if 0:
            return cv2.Canny(img, 100, 200)
        img = cv2.Laplacian(np.float64(img), cv2.CV_64F, None, 3)
        return np.uint8(np.abs(img))

    if use_edges:
        img0, img1 = map(edges, (img0, img1))

    img0_1 = project_image(img0, I0, P0_1, cam1)

    nonnan = np.logical_not(np.isnan(img0_1))
    xs = np.where(np.any(nonnan, axis=0))[0]
    ys = np.where(np.any(nonnan, axis=1))[0]
    xmin0, xmax0, ymin0, ymax0 = xs[0], xs[-1], ys[0], ys[-1]
    xmin1, ymin1 = max(0, xmin0 - margin_px), max(0, ymin0 - margin_px)
    xmax1, ymax1 = min(img1.shape[1], xmax0 + margin_px), min(img1.shape[0], ymax0 + margin_px)
    tcx0, tcy0 = -margin_px, -margin_px

    p_img1 = np.ones((ymax0 - ymin0 + margin_px * 2, xmax0 - xmin0 + margin_px * 2)) * np.nan
    ox, ow = xmin1 - (xmin0 - margin_px), xmax1 - xmin1
    oy, oh = ymin1 - (ymin0 - margin_px), ymax1 - ymin1
    p_img1[oy:oy+oh, ox:ox+ow] = img1[ymin1:ymax1, xmin1:xmax1]
    p_img0 = img0_1[ymin0:ymax0, xmin0:xmax0]

    if 0:
        shape0 = p_img0.shape
        p_img0 = p_img0.flatten()
        I = np.isnan(p_img0)
        p_img0[I] = 0
        mask = np.logical_not(I).astype(np.uint8).reshape(shape0)
        p_img0 = p_img0.astype(np.uint8).reshape(shape0)

        metric = [cv2.TM_SQDIFF, cv2.TM_CCORR_NORMED][1]
        scores = cv2.matchTemplate(p_img1, p_img0, metric, mask=mask)
    else:
        metric = cv2.TM_SQDIFF
        if skip > 1:
            def nanresize(img, sc):
                n = np.isnan(img)
                img.flatten()[n.flatten()] = 0
                sc_img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                sc_nn = cv2.resize(np.logical_not(n).astype(np.uint8), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                with np.errstate(invalid='ignore'):
                    sc_img /= sc_nn         # also puts nans back
                return sc_img
            p_img0, p_img1 = map(lambda x: nanresize(x, 1/skip), (p_img0, p_img1))
        sh, sw, th, tw = p_img1.shape[:2] + p_img0.shape[:2]
        scores = np.ones((sh - th + 1, sw - tw + 1)) * np.nan
        template_match_nb(p_img0, p_img1, scores, 10)

    ij = (np.argmin if metric in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED) else np.argmax)(scores)
    tcyi, tcxi = np.unravel_index(ij, scores.shape)
    tcy, tcx = tcyi * skip + tcy0, tcxi * skip + tcx0

    if 0:
        c = 1 if depthmap else 255
        p_img0 = (np.float64(p_img0) / c) ** (1 / 4 if use_edges else 1)
        p_img1 = (np.float64(img1) / c) ** (1 / 4 if use_edges else 1)
        p0 = np.nanquantile(p_img0, (0.01, 0.99))
        p1 = np.nanquantile(p_img1, (0.01, 0.99))
        vmin, vmax = min(p0[0], p1[0]), max(p0[1], p1[1])

        fig, axs = plt.subplots(2, 2)
        axs = axs.flatten()
        axs[0].imshow(p_img0, vmin=vmin, vmax=vmax)
        axs[0].plot([p_img0.shape[1]/2], [p_img0.shape[0]/2], 'x')
        axs[1].imshow(p_img1, vmin=vmin, vmax=vmax)
        axs[1].plot([xmin0 + p_img0.shape[1]/2 + tcx], [ymin0 + p_img0.shape[0]/2 + tcy], 'x')
        axs[1].plot([xmin0 + p_img0.shape[1]/2], [ymin0 + p_img0.shape[0]/2], 'o', mfc='none')
        axs[2].imshow((scores))  # - np.nanmin(scores))) #  / (np.max(res) - np.min(res)))
        axs[2].plot([tcxi], [tcyi], 'C1x')
        plt.show()

    # get metric displacement from pixel displacement
    r = np.nanmedian(P0_1[:, 2])
    tr_vect = r * np.array([-tcx, -tcy, 0]) / np.array([cam1.matrix[0, 0], cam1.matrix[1, 1], 1])

    return tr_vect, quaternion.one, None


def maybe_decorate(dec_str, condition):
    dec = eval(dec_str)

    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator


@maybe_decorate('nb.njit(nogil=True, parallel=False, cache=False)', nb is not None)
def template_match_nb(templ, img, scores, max_err):
    th, tw = templ.shape[:2]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            sqrd = ((img[i:i + th, j:j + tw] - templ) ** 2).flatten()  # squared difference
            sqrd[sqrd > max_err ** 2] = max_err ** 2
            scores[i, j] = np.nanmean(sqrd)


def show_all_pairs(aflow_path, img_path, image_db):
    for fname in os.listdir(aflow_path):
        if fname[-10:] == '.aflow.png':
            aflow = load_aflow(os.path.join(aflow_path, fname))
            id0, id1 = map(int, fname[:-10].split('_'))
            img0 = cv2.imread(os.path.join(img_path, image_db[id0]), cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(os.path.join(img_path, image_db[id1]), cv2.IMREAD_UNCHANGED)
            show_pair(img0, img1, aflow, image_db[id0], image_db[id1])


def read_raw_img(path, bands, gdtype=None, ndtype=np.float32, gamma=1.0, disp_dir=None,
                 metadata_type='esa/jaxa', q_wxyz=True, crop=None):
    from osgeo import gdal

    gdtype = gdal.GDT_Float32 if gdtype is None else gdtype
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
    img, (bot_v, top_v) = preprocess_image(data, gamma)

    handle = None
    if isinstance(metadata_type, Callable):
        allmeta, metadata, m_disp_dir = metadata_type(path)
    elif not metadata_type:
        allmeta, metadata, m_disp_dir = {}, {}, None
    else:
        allmeta, metadata, m_disp_dir = parse_metadata(path, q_wxyz)

    ele = {
        'forward': 'img = np.clip(255*((raw_img - bg) / max_val) ** (1 / gamma) + 0.5, 0, 255).astype(np.uint8)',
        'backward': 'raw_img = max_val * (img/255) ** gamma + bg',
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
    import pvl

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


def calc_target_pose(xyz, cam, sc_ori, ref_north_v):
    try:
        cam_sc_trg_pos, cam_sc_trg_ori = estimate_pose_pnp(cam, xyz)
    except cv2.error as e:
        logging.warning("can't calculate relative pose: %s" % e)
        cam_sc_trg_pos, cam_sc_trg_ori = [None] * 2

    if cam_sc_trg_pos is None or cam_sc_trg_ori is None:
        return sc_ori, None, None

    for _ in range(2):
        _sc_ori = sc_ori or quaternion.one
        sc_trg_pos = q_times_v(_sc_ori, cam_sc_trg_pos)  # to icrf
        trg_ori = _sc_ori * cam_sc_trg_ori  # TODO: verify, how?
        est_north_v = q_times_v(trg_ori, np.array([0, 0, 1]))
        rot_axis_err = math.degrees(angle_between_v(est_north_v, ref_north_v))
        if rot_axis_err > 15 and False:
            # idea is to check if have erroneous sc_ori from image metadata
            # TODO: fix rotations, now the north pole vector rotates slowly around the x-axis (itokawa)
            #  - However, there's currently no impact from bad sc_ori as the relative orientation
            #    is used for image rotation
            sc_ori = None
        else:
            # north.append(est_north_v)
            break

    # in cam frame where axis +x, up +z
    return sc_ori, sc_trg_pos, trg_ori


def safe_split(x, is_q):
    if x is None:
        return (None,) * (4 if is_q else 3)
    return (*x[:3],) if not is_q else (x.w, x.x, x.y, x.z)


def check_img(img, bg_q=0.04, fg_q=240, sat_lo_q=0.998, sat_hi_q=0.9999, fg_lim=50, sat_lim=5, min_side=256):
    if fg_q > 1:
        # fg_q is instead the diameter of a half-circle in px
        fg_q = 1 - 0.5 * np.pi * (fg_q/2)**2 / np.prod(img.shape[:2])

    bg, fg, sat_lo, sat_hi = np.quantile(img, (bg_q, fg_q, sat_lo_q, sat_hi_q))
    return np.min(img.shape) >= min_side and fg - bg >= fg_lim and sat_hi - sat_lo > sat_lim


def write_data(path, img, data, metastr=None, xyzd=False, cam=None):
    cv2.imwrite(path + '.png', img, (cv2.IMWRITE_PNG_COMPRESSION, 9))
    if data is not None and data.size > 0:
        save_xyz(path + '.xyz', data[:, :, :3])
        if data.shape[2] > 3:
            if xyzd:
                save_mono(path + '.d', data[:, :, 3])
            else:
                assert cam is not None, 'cam is None'
                save_mono(path + '.s', data[:, :, 3])
                d = _convert_xyz2d(data[:, :, :3], cam)
                save_mono(path + '.d', d)
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


def keys_to_lower(d):
    return nested_filter(d, lambda x: True, lambda x: x, lambda x: x.lower())


def fix_img_dbs(path):
    from ..tools import if_none_q

    dbfile = 'dataset_all.sqlite'
    eros_path = os.path.join(path, 'eros', dbfile)
    itokawa_path = os.path.join(path, 'itokawa', dbfile)
    osinac_path = os.path.join(path, 'cg67p', 'osinac', dbfile)

    for dbpath in (eros_path, itokawa_path, osinac_path):
        index = ImageDB(dbpath)
        values = []
        for id, file, sc_qw, sc_qx, sc_qy, sc_qz, sc_trg_x, sc_trg_y, sc_trg_z, \
                      trg_qw, trg_qx, trg_qy, trg_qz in index.get_all((
                          'id', 'file', 'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz', 'sc_trg_x', 'sc_trg_y', 'sc_trg_z',
                          'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz')):
            sc_q = if_none_q(sc_qw, sc_qx, sc_qy, sc_qz, fallback=quaternion.one)
            trg_q = if_none_q(trg_qw, trg_qx, trg_qy, trg_qz, fallback=np.quaternion(*[np.nan]*4))
            trg_q = sc_q * from_opencv_q(sc_q.conj() * trg_q)
            sc_trg_v = np.array([sc_trg_x, sc_trg_y, sc_trg_z])
            if dbpath is not osinac_path:
                sc_trg_v = q_times_v(sc_q, from_opencv_v(q_times_v(sc_q.conj(), sc_trg_v)))
            values.append((id, file, *trg_q.components, *sc_trg_v))
        index.set(('id', 'file', 'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz', 'sc_trg_x', 'sc_trg_y', 'sc_trg_z'), values)


class DisableLogger:
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)


if __name__ == '__main__':
    if 0:
        create_image_pairs_script()
    else:
        convert_from_exr_to_png()

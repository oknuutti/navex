import math
import os
import argparse
import logging

from tqdm import tqdm
import numpy as np

from navex.datasets.preproc.tools import load_xyzs, calc_aflow
from navex.datasets.tools import ImageDB, angle_between_v, save_aflow, tf_view_unit_v, Camera

# # camera params from osiris-rex tagcams
# #   https://sbnarchive.psi.edu/pds4/orex/orex.tagcams/document/tagcams_inst_desc.pdf
# # sensor datasheet
# #   https://www.onsemi.com/pdf/datasheet/mt9p031-d.pdf
# # lens reference
# #   http://www.msss.com/brochures/xfov.pdf
ow, oh, sc = 2592, 1944, 0.3951
resolution = (int(sc * ow), int(sc * oh))
# CAM = Camera(resolution=resolution, center=((ow - 1) * sc / 2, (oh - 1) * sc / 2),
#              pixel_size=2.2e-6 / sc, focal_length=7.7e-3, f_num=3.5)
#
# turns out the above didn't match exactly the cam matrix used during data generation:
CAM = Camera(resolution=resolution,
             matrix=np.array([[1.26724447e+03, 0.00000000e+00, 5.12000000e+02],
                              [0.00000000e+00, 1.33916715e+03, 3.84000000e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))


def raw_synth():
    """
    read synthetic images created with https://github.com/oknuutti/visnav-py
    """
    parser = argparse.ArgumentParser('Process synthetic images')

    parser.add_argument('--src', help="input folder")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--index', default='dataset_all.sqlite',
                        help="index file name in the output folder")
    parser.add_argument('--pairs', default='pairs.txt', help="pairing file to create in root")
    parser.add_argument('--aflow', default='aflow', help="subfolder where the aflow files are generated")

    # parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    # parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")
    # parser.add_argument('--img-max', type=int, default=3, help="how many times same images can be repated in pairs")
    # parser.add_argument('--min-angle', type=float, default=5,
    #                     help="min angle (deg) on the unit sphere for pair creation")
    # parser.add_argument('--max-angle', type=float, default=20,
    #                     help="max angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--min-matches', type=int, default=10000,
                        help="min pixel matches in order to approve generated pair")
    # parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.dst, args.index)

    logging.info('building the index file...')
    with open(os.path.join(args.src, 'dataset_all.txt'), 'r') as fh:
        rows = []
        pairs = {}
        dists = {}
        coords = {}

        s = 0
        for k, line in enumerate(tqdm(fh)):
            row = line.split(' ')
            if len(row) != 18:
                if k > 3:
                    logging.warning('too few cells detected on line %d: %s' % (k, line))
                else:
                    s = k
                continue

            id = k - s
            _, a, b, fname, *floats = row
            sc_trg_x, sc_trg_y, sc_trg_z, sc_qw, sc_qx, sc_qy, sc_qz, \
                    trg_qw, trg_qx, trg_qy, trg_qz, sc_sun_x, sc_sun_y, sc_sun_z = map(float, floats)

            rand = np.random.uniform(0, 1)
            rows.append((id, fname, rand, sc_trg_x, sc_trg_y, sc_trg_z, sc_qw, sc_qx, sc_qy, sc_qz,
                         trg_qw, trg_qx, trg_qy, trg_qz, sc_sun_x, sc_sun_y, sc_sun_z))

            if a not in pairs:
                pairs[a] = []
            pairs[a].append((b, id, fname))

            dists[id] = np.linalg.norm([sc_trg_x, sc_trg_y, sc_trg_z])
            coords[id] = tf_view_unit_v((np.quaternion(sc_qw, sc_qx, sc_qy, sc_qz).conj() *
                                         np.quaternion(trg_qw, trg_qx, trg_qy, trg_qz)))

    index = ImageDB(index_path, truncate=True)
    index.add(('id', 'file', 'rand', 'sc_trg_x', 'sc_trg_y', 'sc_trg_z', 'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz',
               'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz', 'sc_sun_x', 'sc_sun_y', 'sc_sun_z'), rows)

    logging.info('calculating aflow...')

    aflow_path = os.path.join(args.dst, args.aflow)
    os.makedirs(aflow_path, exist_ok=True)
    pairs_path = os.path.join(args.dst, args.pairs)
    handled_ids = set()

    if not os.path.exists(pairs_path):
        with open(pairs_path, 'w') as fh:
            fh.write('image_id_0 image_id_1 sc_diff angle_diff match_ratio matches\n')
    else:
        with open(pairs_path, 'r') as fh:
            for line in fh:
                try:
                    c = line.split(' ')
                    handled_ids.add(int(c[0]))
                    handled_ids.add(int(c[1]))
                except:
                    pass

    hz_fov, pbar, add_count = 44, tqdm(pairs.values(), mininterval=3), 0
    for tot, pair in enumerate(pbar):
        pair = sorted(pair, key=lambda x: x[0])
        (_, id0, fname0), (_, id1, fname1) = pair
        if id0 in handled_ids and id1 in handled_ids:
            continue

        xyzs0 = load_xyzs(os.path.join(args.src, fname0), hz_fov=hz_fov)
        xyzs1 = load_xyzs(os.path.join(args.src, fname1), hz_fov=hz_fov)

        # calculate pair stats
        angle = math.degrees(angle_between_v(coords[id0], coords[id1]))
        sc_diff = max(dists[id0] / dists[id1], dists[id1] / dists[id0])
        max_matches = min(np.sum(np.logical_not(np.isnan(xyzs0[:, :, 0]))),
                          np.sum(np.logical_not(np.isnan(xyzs1[:, :, 0]))))

        ratio, matches = np.nan, -1
        if max_matches > args.min_matches:
            aflow = calc_aflow(xyzs0, xyzs1)
            matches = np.sum(np.logical_not(np.isnan(aflow[:, :, 0])))

            if matches > args.min_matches:
                ratio = matches / max_matches
                save_aflow(os.path.join(aflow_path, '%d_%d.aflow.png' % (id0, id1)), aflow)
                add_count += 1

        with open(pairs_path, 'a') as fh:
            fh.write('%d %d %.3f %.3f %.3f %d\n' % (id0, id1, sc_diff, angle, ratio, matches))

        pbar.set_postfix({'added': add_count, 'ratio': add_count/(tot + 1)}, refresh=False)


if __name__ == '__main__':
    raw_synth()

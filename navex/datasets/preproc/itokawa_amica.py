import math
import os
import gzip
import shutil
import argparse
import logging

from tqdm import tqdm
import numpy as np
import quaternion

from navex.datasets.preproc.tools import read_raw_img, write_data, safe_split, create_image_pairs, check_img, \
    relative_pose
from navex.datasets.tools import ImageDB, find_files, Camera, q_times_v, angle_between_v, spherical2cartesian, eul_to_q, \
    plot_vectors


def raw_itokawa():
    """
    read amica images of hayabusa 1 to itokawa, instrument details:
     - https://arxiv.org/ftp/arxiv/papers/0912/0912.4797.pdf
     - main info is fov: 5.83° x 5.69°, focal length: 120.80 mm, px size 12 um, active pixels 1024x1000,
       zero level monitoring with 12px left & 12px right, 12-bits, eff aperture 15 mm, full well 70ke-,
       gain factor: 17 DN/e-, readout noise 60e-

    info on data format: https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    """
    parser = argparse.ArgumentParser('Process data from Hayabusa about Itokawa')

    parser.add_argument('--src', help="input folder")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--index', default='dataset_all.sqlite',
                        help="index file name in the output folder")
    parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")

    parser.add_argument('--pairs', default='pairs.txt', help="pairing file to create in root")
    parser.add_argument('--aflow', default='aflow', help="subfolder where the aflow files are generated")
    parser.add_argument('--img-max', type=int, default=3, help="how many times same images can be repated in pairs")
    parser.add_argument('--min-angle', type=float, default=5,
                        help="min angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--max-angle', type=float, default=20,
                        help="max angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--min-matches', type=int, default=10000,
                        help="min pixel matches in order to approve generated pair")
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    cam = Camera(resolution=(1024, 1024), center=(511.5, 511.5), pixel_size=1.2e-5, focal_length=0.1208, f_num=8.0)
    north_ra, north_dec = math.radians(90.53), math.radians(-66.30)
    ref_north_v = spherical2cartesian(north_dec, north_ra, 1)
#    meta2icrf_q = quaternion.one
#    meta2icrf_q = eul_to_q((-np.pi/2,), 'y')
#    meta2icrf_q = eul_to_q((np.pi, -np.pi/2), 'yz')

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.dst, args.index)

    if not os.path.exists(index_path):
        logging.info('building the index file by scanning ftp server for image files...')

        files = find_files(args.src, ext='.lbl', relative=True)
        rows = []
        # north = []
        pbar, n_ok, tot = tqdm(files), 0, 0
        for i, fname in enumerate(pbar):
            path = os.path.join(args.src, fname[:-4])

            with gzip.open(path + '.img.gz', 'rb') as fh_in:
                with open(path + '.img', 'wb') as fh_out:
                    shutil.copyfileobj(fh_in, fh_out)

            img, data, metadata, metastr = read_itokawa_img(path + '.lbl')
            # if metadata['sc_ori']:
            #     metadata['sc_ori'] = metadata['sc_ori'] * meta2icrf_q

            cam_sc_trg_pos, cam_sc_trg_ori = relative_pose(data[:, :, :3], cam)
            for _ in range(2):
                sc_ori = metadata['sc_ori'] or quaternion.one
                sc_trg_pos = q_times_v(sc_ori, cam_sc_trg_pos)     # to icrf
                trg_ori = sc_ori * cam_sc_trg_ori  # TODO: verify, how?
                est_north_v = q_times_v(trg_ori, np.array([0, 0, 1]))
                rot_axis_err = math.degrees(angle_between_v(est_north_v, ref_north_v))
                if rot_axis_err > 15 and False:
                    # idea is to check if have erroneous sc_ori from image metadata
                    # TODO: fix rotations, now the north pole vector rotates slowly around the x-axis
                    #  - However, there's currently no impact from bad sc_ori as the relative orientation
                    #    is used for image rotation
                    metadata['sc_ori'] = None
                else:
                    # north.append(est_north_v)
                    break

            ok = check_img(img, fg_q=150, sat_lo_q=0.995)
            rand = np.random.uniform(0, 1) if ok else -1
            rows.append((i, fname[:-4] + '.png', rand) + safe_split(metadata['sc_ori'], True)
                        + safe_split(sc_trg_pos, False) + safe_split(trg_ori, True))

            if ok or args.debug:
                write_data(os.path.join(args.dst, fname[:-4]) + ('' if ok else ' - FAILED'), img, data, metastr)

            os.unlink(path + '.img')

            tot += 1
            n_ok += 1 if ok else 0
            pbar.set_postfix({'images ok': '%.1f%%' % (100 * n_ok/tot)}, refresh=False)

        # plot_vectors(np.stack(north))
        index = ImageDB(index_path, truncate=True)
        index.add(('id', 'file', 'rand', 'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz',
                   'sc_trg_x', 'sc_trg_y', 'sc_trg_z', 'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz'), rows)
    else:
        index = ImageDB(index_path)

    fov_hz = 5.83
    create_image_pairs(args.dst, index, args.pairs, '', args.aflow, args.img_max, fov_hz,
                       args.min_angle, args.max_angle, args.min_matches, read_meta=True, start=args.start,
                       end=args.end)


def read_itokawa_img(path):
    img, data, metadata, metastr = read_raw_img(path, (1, 2, 3, 4, 11, 12), disp_dir=('up', 'right'), q_wxyz=True)

    # select only pixel value and x, y, z; calculate pixel size by taking max of px
    # - for band indexes, see https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    px_size = np.atleast_3d(np.max(data[:, :, 3:5], axis=2))
    data = np.concatenate((data[:, :, 0:3], px_size), axis=2)
    data[data <= -1e30] = np.nan

    return img, data, metadata, metastr


if __name__ == '__main__':
    raw_itokawa()

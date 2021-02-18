import os
import gzip
import shutil
import argparse
import logging

from tqdm import tqdm
import numpy as np

from navex.datasets.preproc.tools import read_raw_img, write_data, safe_split, create_image_pairs
from navex.datasets.tools import ImageDB, find_files


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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.dst, args.index)

    if not os.path.exists(index_path):
        logging.info('building the index file by scanning ftp server for image files...')

        files = find_files(args.src, ext='.lbl', relative=True)
        rows = []
        for i, fname in enumerate(tqdm(files)):
            path = os.path.join(args.src, fname[:-4])
            extracted = False

            if not os.path.exists(path + '.img'):
                extracted = True
                with gzip.open(path + '.img.gz', 'rb') as fh_in:
                    with open(path + '.img', 'wb') as fh_out:
                        shutil.copyfileobj(fh_in, fh_out)

            img, data, metadata = read_itokawa_img(path + '.lbl')
            write_data(os.path.join(args.dst, fname[:-4]), img, data)
            rows.append((i, fname[:-4] + '.png') + safe_split(metadata['sc_ori'], True))

            if extracted:
                os.unlink(path + '.img')

        index = ImageDB(index_path, truncate=True)
        index.add(('id', 'file', 'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz'), rows)
    else:
        index = ImageDB(index_path)

    fov_hz = 5.83
    create_image_pairs(args.dst, index, args.pairs, '', args.aflow, args.img_max, fov_hz,
                       args.min_angle, args.max_angle, args.min_matches, read_meta=True, start=args.start,
                       end=args.end)


def read_itokawa_img(path):
    img, data, metadata = read_raw_img(path, (1, 2, 3, 4, 11, 12))

    # select only pixel value and x, y, z; calculate pixel size by taking max of px
    # - for band indexes, see https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    px_size = np.atleast_3d(np.max(data[:, :, 3:5], axis=2))
    data = np.concatenate((data[:, :, 0:3], px_size), axis=2)
    data[data <= -1e30] = np.nan

    return img, data, metadata


if __name__ == '__main__':
    raw_itokawa()

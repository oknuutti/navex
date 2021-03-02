import argparse
import ftplib
import math
import os
import re
import shutil
import random
import time
import gzip
import logging

from tqdm import tqdm
import numpy as np

from navex.datasets.tools import ImageDB, find_files
from navex.datasets.preproc.tools import write_data, read_raw_img, create_image_pairs, safe_split, check_img


def main():
    parser = argparse.ArgumentParser('Download and process data from Rosetta about C-G/67P')
    # parser.add_argument('--host', default='psa.esac.esa.int',
    #                     help="ftp host from which to fetch the data")
    parser.add_argument('--src', help="path with the data")
    # parser.add_argument('--regex', help="at given path, select folders/files based on this")
    # parser.add_argument('--deep-path', help="path to follow after regex match to arrive at the data")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--index', default='dataset_all.sqlite',
                        help="index file name in the output folder")
    parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")

    parser.add_argument('--pairs', default='pairs.txt', help="pairing file to create in root")
    parser.add_argument('--aflow', default='aflow', help="subfolder where the aflow files are generated")
    # parser.add_argument('--instr', choices=('navcam', 'osinac', 'osiwac'),
    #                     help="which instrument, navcam, osinac or osiwac?")
    # parser.add_argument('--has-lbl', type=int, default=-1, help="src has separate lbl files")
    # parser.add_argument('--has-geom', type=int, default=-1, help="img data has geometry backplanes")
    # parser.add_argument('--fov', type=float, help="horizontal field of view in degrees")
    parser.add_argument('--check-img', type=int, default=0, help="try to screen out bad images")
    parser.add_argument('--img-max', type=int, default=3, help="how many times same images can be repated in pairs")
    parser.add_argument('--min-angle', type=float, default=0,
                        help="min angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--max-angle', type=float, default=0,
                        help="max angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--min-matches', type=int, default=10000,
                        help="min pixel matches in order to approve generated pair")

    args = parser.parse_args()

    fov = 2.9   # 2.9 x 2.25 deg

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.dst, args.index)
    progress_path = os.path.join(args.dst, 'progress.txt')

    if os.path.exists(progress_path):
        with open(progress_path) as fh:
            archives_done = {line.strip() for line in fh}
    else:
        archives_done = set()

    index = ImageDB(index_path, truncate=not os.path.exists(index_path))
    files = [(id, file) for id, file in index.get_all(('id', 'file',))]

    next_id = (0 if len(files) else np.max([id for id, _ in files])) + 1

    # find all archives from src, dont include if in archives_done
    # TODO
    archives = []

    # process archives in order
    for archive in tqdm(archives, desc='archives'):
        if not os.path.exists(os.path.join(args.dst, archive)):
            # TODO: download
            pass

        # TODO: extract files

        # process files one by one
        arch_files = find_files(os.path.join(args.dst, 'nearmsi.shapebackplane'), ext='.xml')
        for fullpath in tqdm(arch_files, desc='files'):
            process_file(fullpath, args.dst, next_id, index, args)
            next_id += 1

        # TODO: remove extracted dir

        with open(progress_path, 'a') as fh:
            fh.write(archive + '\n')

    create_image_pairs(args.dst, index, args.pairs, args.dst, args.aflow, args.img_max, fov,
                       args.min_angle, args.max_angle, args.min_matches, read_meta=True, start=args.start,
                       end=args.end)


def process_file(src_path, dst_path, id, index, args):
    dst_file = os.path.join(*os.path.split(src_path)[-2:])[-4:] + '.png'
    dst_path = os.path.join(dst_path, dst_file)
    if not os.path.exists(dst_path):
        tmp_file = dst_path[:-4]
        os.makedirs(os.path.dirname(tmp_file), exist_ok=True)

        # extract *.fit.gz
        extracted = False
        if not os.path.exists(tmp_file + '.fit'):
            extracted = True
            with gzip.open(tmp_file + '.fit.gz', 'rb') as fh_in:
                with open(tmp_file + '.fit', 'wb') as fh_out:
                    shutil.copyfileobj(fh_in, fh_out)

        img, data, metastr, metadata = read_eros_img(tmp_file + '.xml')

        os.unlink(tmp_file + '.xml')
        os.unlink(tmp_file + '.fit.gz')
        if extracted:
            os.unlink(tmp_file + '.fit')

        ok = True
        if args.check_img:
            ok = check_img(img)

        if 1 or ok:
            write_data(tmp_file + ('-fail' if not ok else ''), img, data, metastr, xyzd=False)
            ok = True

        if ok:
            index.add(('id', 'file'), [(id, dst_file)])


def read_eros_img(path):
    img, data, metastr, metadata = read_raw_img(path, (1, 2, 3, 4, 11, 12),
                                                disp_dir=('down', 'left'), gamma=1.8, q_wxyz=False)

    # select only pixel value, model x, y, z and depth
    # - for band indexes, see https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    px_size = np.atleast_3d(np.max(data[:, :, 3:5], axis=2))
    data = np.concatenate((data[:, :, 0:3], px_size), axis=2)
    data[data <= -1e30] = np.nan    # TODO: debug this

    return img, data, metastr, metadata


if __name__ == '__main__':
    main()

import argparse
import ftplib
import math
import os
import re
import shutil
import random
import tarfile
import time
import gzip
import logging
import requests
from bs4 import BeautifulSoup

from tqdm import tqdm
import numpy as np

from navex.datasets.tools import ImageDB, find_files, find_files_recurse
from navex.datasets.preproc.tools import write_data, read_raw_img, create_image_pairs, safe_split, check_img, get_file


def main():
    parser = argparse.ArgumentParser('Download and process data from NEAR MSI about Eros')
    # parser.add_argument('--host', default='psa.esac.esa.int',
    #                     help="ftp host from which to fetch the data")
    parser.add_argument('--src', default="https://sbnarchive.psi.edu/pds4/near/nearmsi_shapebackplane_downloads/",
                        help="path with the data")
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

    next_id = (0 if len(files) == 0 else np.max([id for id, _ in files])) + 1

    # find all archives from src, dont include if in archives_done
    page = requests.get(args.src)  # , verify=False)
    soup = BeautifulSoup(page.content, 'html.parser')
    archives = [a['href']
                for a in soup.find_all(name="a")
                if re.match(r'^nearmsi\..*?\.tar\.gz$', a['href'])
                and a['href'] not in archives_done]

    # process archives in order
    pbar = tqdm(archives, desc='archives')
    tot, n_add, n_ok = 0, 0, 0
    for archive in pbar:
        archive_url = args.src + '/' + archive
        archive_path = os.path.join(args.dst, archive)
        if not os.path.exists(archive_path):
            get_file(archive_url, archive_path)

        # extract archive
        extract_path = os.path.join(args.dst, 'tmp')
        tar = tarfile.open(archive_path, "r:gz")
        tar.extractall(extract_path)
        tar.close()

        # process files one by one
        arch_files = find_files_recurse(extract_path, ext='.xml')
        for fullpath in tqdm(arch_files, desc='files', mininterval=3):
            added, ok = process_file(fullpath, args.dst, next_id, index, args)
            next_id += 1
            tot += 1
            n_add += 1 if added else 0
            n_ok += 1 if ok else 0

        # remove extracted dir and downloaded archive
        shutil.rmtree(extract_path)
        os.unlink(archive_path)

        with open(progress_path, 'a') as fh:
            fh.write(archive + '\n')

        pbar.set_postfix({'added': '%.1f%%' % (100 * n_add/tot), 'images ok': '%.1f%%' % (100 * n_ok/tot)}, refresh=False)

    create_image_pairs(args.dst, index, args.pairs, args.dst, args.aflow, args.img_max, fov,
                       args.min_angle, args.max_angle, args.min_matches, read_meta=True, start=args.start,
                       end=args.end)


def process_file(src_path, dst_path, id, index, args):
    src_path = src_path[:-4]
    parts = os.path.normpath(src_path).split(os.sep)[-2:]
    dst_file = os.path.join(*parts) + '.png'
    dst_path = os.path.join(dst_path, dst_file)
    if not os.path.exists(dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # extract *.fit.gz
        extracted = False
        if not os.path.exists(src_path + '.fit'):
            extracted = True
            with gzip.open(src_path + '.fit.gz', 'rb') as fh_in:
                with open(src_path + '.fit', 'wb') as fh_out:
                    shutil.copyfileobj(fh_in, fh_out)

        img, data = read_eros_img(src_path + '.fit')

        os.unlink(src_path + '.xml')
        os.unlink(src_path + '.fit.gz')
        if extracted:
            os.unlink(src_path + '.fit')

        ok = True
        if args.check_img:
            ok = check_img(img, lo_q=0.01, hi_q=0.90)

        added = False
        if ok:
            rand = np.random.uniform(0, 1)
            index.add(('id', 'file', 'rand'), [(id, dst_file, rand)])

            if args.start <= rand < args.end:
                write_data(dst_path[:-4], img, data, xyzd=False)
                added = True

        return added, ok
    return True, True


def read_eros_img(path):
    # this eros data doesn't have any interesting metadata, would need to use the spice kernels
    img, data, _, _ = read_raw_img(path, (1, 2, 3, 4, 11, 12), metadata_type=None,
                                                disp_dir=('down', 'right'), gamma=1.8, q_wxyz=False)

    # crop out sides, which seem to be empty / have some severe artifacts/problems
    img = img[:, 15:-14]
    data = data[:, 15:-14, :]

    # select only pixel value, model x, y, z and depth
    # - for band indexes, see https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    px_size = np.atleast_3d(np.max(data[:, :, 3:5], axis=2))
    data = np.concatenate((data[:, :, 0:3], px_size), axis=2)
    data[data <= -1e30] = np.nan    # TODO: debug this

    return img, data


if __name__ == '__main__':
    main()

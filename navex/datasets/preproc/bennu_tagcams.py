import argparse
import json
import math
import os
import re
import shutil
import logging
import zipfile

import requests
from bs4 import BeautifulSoup

from tqdm import tqdm
import numpy as np
import pds4_tools as pds4

from navex.datasets.tools import ImageDB, find_files, find_files_recurse
from navex.datasets.preproc.tools import write_data, read_raw_img, create_image_pairs, safe_split, check_img, get_file, \
    ypr_to_q, DisableLogger, keys_to_lower


def main():
    DEBUG = False

    parser = argparse.ArgumentParser('Download and process data from OSIRIS-REx TAGCAMs about Bennu')
    parser.add_argument('--src', default="https://sbnarchive.psi.edu/pds4/orex/downloads_tagcams/",
                        help="path with the data")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--index', default='dataset_all.sqlite',
                        help="index file name in the output folder")
    parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")

    args = parser.parse_args()
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
    phases = [
        'detailed_survey',
        'orbit_a',
        'orbit_b',
        'orbit_c',
        'orbit_r',
        'preliminary_survey',
        'recon',
        'recon_b',
        'recon_c',
        'sample_collection',
    ]

    if DEBUG:
        phases = ['orbit_r']

    archives = [a['href']
                for a in soup.find_all(name="a")
                if re.match('^tagcams_data_raw_(' + ('|'.join(phases)) + r')\.zip$', a['href'])
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
        if not DEBUG:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

        # process files one by one
        arch_files = find_files_recurse(extract_path, ext='.fits')
        for fullpath in tqdm(arch_files, desc='files', mininterval=3):
            added, ok = process_file(fullpath, args.dst, next_id, index, args)
            next_id += 1
            tot += 1
            n_add += 1 if added else 0
            n_ok += 1 if ok else 0

        # remove extracted dir and downloaded archive
        if not DEBUG:
            shutil.rmtree(extract_path)
            os.unlink(archive_path)

        with open(progress_path, 'a') as fh:
            fh.write(archive + '\n')

        pbar.set_postfix({'added': '%.1f%%' % (100 * n_add/tot), 'images ok': '%.1f%%' % (100 * n_ok/tot)}, refresh=False)


def process_file(src_path, dst_path, id, index, args):
    src_path, ext = os.path.splitext(src_path)
    parts = os.path.normpath(src_path).split(os.sep)[-2:]
    dst_file = os.path.join(*parts) + '.png'
    dst_path = os.path.join(dst_path, dst_file)
    if not os.path.exists(dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        img, metadata, metastr = read_bennu_img(src_path + ext)
        ok = check_img(img, lo_q=0.05, hi_q=0.99, sat_q=0.999)

        added = False
        if ok:
            rand = np.random.uniform(0, 1)
            index.set(('id', 'file', 'rand', 'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz',
                       'sc_sun_x', 'sc_sun_y', 'sc_sun_z',
                       'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz'),
                      [(id, dst_file, rand)
                      + safe_split(metadata['sc_ori'], True)
                      + safe_split(metadata['sc_sun_pos'], False)
                      + safe_split(metadata['trg_ori'], True)])

            if args.start <= rand < args.end:
                write_data(dst_path[:-4], img, None, metastr=metastr)
                added = True

        os.unlink(src_path + '.xml')
        os.unlink(src_path + '.fits')
        return added, ok
    return True, True


def read_bennu_img(path):
    # this eros data doesn't have any interesting metadata, would need to use the spice kernels
    img, _, metastr, metadata = read_raw_img(path, (1,), disp_dir=('up', 'right'),
                                             metadata_type=parse_bennu_metadata, gamma=1.8, q_wxyz=True)

    return img, metadata, metastr


def parse_bennu_metadata(path):
    # return metadata in J2000 frame
    path, _ = os.path.splitext(path)

    with DisableLogger():
        meta = keys_to_lower(pds4.read(path + '.xml').label.to_dict())

    try:
        mm = meta['product_observational']['observation_area']['mission_area']
        mms = mm['orex:spatial']

        # TODO: check if correct
        trg_nca = float(mms['orex:bennana'])
        trg_ra = float(mms['orex:bennu_ra'])
        trg_dec = float(mms['orex:bennu_dec'])
        trg_ori_q = ypr_to_q(math.radians(trg_dec), math.radians(trg_ra), -math.radians(trg_nca))

        # TODO: check if need '-' or not
        sc_sun_pos_v = -np.array([mms['orex:sunbeamx'], mms['orex:sunbeamy'], mms['orex:sunbeamz']]).astype(np.float32)

        if 'orex:tagcams_instrument_attributes' in mm:
            mmi = mm['orex:tagcams_instrument_attributes']
        else:
            mmi = mm['orex:tcm_instrument_attributes']
        sc_ori_q = np.quaternion(*np.array([mmi['orex:quaternion0'], mmi['orex:quaternion1'],
                                  mmi['orex:quaternion2'], mmi['orex:quaternion3']]).astype(np.float32))
    except KeyError as e:
        raise Exception('KeyError for %s! Meta:\n%s' % (path, json.dumps(meta, sort_keys=True, indent=4),)) from e

    # bennu data had this, would need to use spice, I suppose
    sc_trg_pos_v = None

    metadata = {'sc_ori': sc_ori_q, 'sc_sun_pos': sc_sun_pos_v, 'trg_ori': trg_ori_q, 'sc_trg_pos': sc_trg_pos_v}
    return meta, metadata, None


if __name__ == '__main__':
    main()

import argparse
import ftplib
import math
import os
import re
import shutil
import random
import time
import logging

from tqdm import tqdm
import numpy as np

from navex.datasets.tools import ImageDB
from navex.datasets.preproc.tools import write_data, read_raw_img, create_image_pairs, safe_split

FOV = {
    'navcam': 5,
    'osinac': 2.21,
    'osiwac': 0,  # TODO: check these
}


def main():
    parser = argparse.ArgumentParser('Download and process data from Rosetta about C-G/67P')
    parser.add_argument('--host', default='psa.esac.esa.int',
                        help="ftp host from which to fetch the data")
    parser.add_argument('--src', default='/pub/mirror/INTERNATIONAL-ROSETTA-MISSION/OSINAC',
                        help="path with the data")
    parser.add_argument('--regex', default=r'^RO-C-OSINAC-5-.*',
                        help="at given path, select folders/files based on this")
    parser.add_argument('--dst', help="output folder")
    parser.add_argument('--index', default='dataset_all.sqlite',
                        help="index file name in the output folder")
    parser.add_argument('--start', type=float, default=0.0, help="where to start processing [0-1]")
    parser.add_argument('--end', type=float, default=1.0, help="where to stop processing [0-1]")

    parser.add_argument('--pairs', default='pairs.txt', help="pairing file to create in root")
    parser.add_argument('--aflow', default='aflow', help="subfolder where the aflow files are generated")
    parser.add_argument('--instr', choices=('navcam', 'osinac', 'osiwac'),
                        help="which instrument, navcam, osinac or osiwac?")
    parser.add_argument('--img-max', type=int, default=3, help="how many times same images can be repated in pairs")
    parser.add_argument('--min-angle', type=float, default=0,
                        help="min angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--max-angle', type=float, default=0,
                        help="max angle (deg) on the unit sphere for pair creation")
    parser.add_argument('--min-matches', type=int, default=10000,
                        help="min pixel matches in order to approve generated pair")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.dst, args.index)
    src_path = args.src.split('/')

    ftp = ftplib.FTP(args.host)
    ftp.login()

    if not os.path.exists(index_path):
        logging.info('building the index file by scanning ftp server for image files...')

        files = []
        scan_ftp(ftp, src_path, [args.regex, 'data', 'img'], files)

        index = ImageDB(index_path, truncate=True)
        files = sorted(['/'.join((fpath[-4], fpath[-1])) for fpath in files])
        index.add(('id', 'file'), [(i, fname[:-4] + '.png') for i, fname in enumerate(files)])
    else:
        index = ImageDB(index_path)

    # read index file
    files = []
    for id, fname, sc_qw in index.get_all(('id', 'file', 'sc_qw'), start=args.start, end=args.end):
        tmp = fname.split('/')
        files.append((id, src_path + [tmp[0], 'DATA', 'IMG', tmp[1][:-4] + '.IMG'], fname, sc_qw))

    logging.info('%d/%d files selected for processing...' % (len(files), len(index)))
    for id, src_file, dst_file, sc_qw in tqdm(files):
        dst_path = os.path.join(args.dst, dst_file)
        if not os.path.exists(dst_path) or sc_qw is None:
            tmp_file = dst_path[:-4] + '.IMG'
            os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
            for i in range(10):
                try:
                    with open(tmp_file, 'wb') as fh:
                        ftp.retrbinary("RETR " + '/'.join(src_file), fh.write)
                    break
                except Exception as e:
                    assert i < 9, 'Failed to download %s 10 times due to %s, giving up' % (src_file, e)
                    logging.warning('Got exception %s while downloading %s' % (e, src_file))
                    logging.warning('Trying again in 10s (#%d)' % (i+1,))
                    time.sleep(10)
                    ftp.close()
                    ftp = ftplib.FTP(args.host)
                    ftp.login()

            img, data, metastr, metadata = read_cg67p_img(tmp_file)
            write_data(tmp_file[:-4], img, data, metastr, xyzd=True)
            index.set(('id', 'file', 'sc_qw', 'sc_qx', 'sc_qy', 'sc_qz',
                       'sc_sun_x', 'sc_sun_y', 'sc_sun_z',
                       'sc_trg_x', 'sc_trg_y', 'sc_trg_z'),
                      [(id, dst_file)
                      + safe_split(metadata['sc_ori'], True)
                      + safe_split(metadata['sc_sun_pos'], False)
                      + safe_split(metadata['sc_trg_pos'], False)])
            os.unlink(tmp_file)
    ftp.close()

    create_image_pairs(args.dst, args.index, args.pairs, args.dst, args.aflow, args.img_max, FOV[args.instr],
                       args.min_angle, args.max_angle, args.min_matches, read_meta=True, start=args.start,
                       end=args.end)


def scan_ftp(ftp, path, filter, files, depth=0):
    ftp.cwd('/'.join(path))

    dirs = []
    entries = []
    ftp.dir(entries.append)
    for entry in entries:
        name = entry.split(' ')[-1].strip()
        if len(filter) == 0 or re.match(filter[0], name, re.IGNORECASE):
            if entry[0] == "d":
                dirs.append(name)
            elif name[-4:] == '.IMG':
                files.append(path + [name])

    if depth == 0:
        dirs = tqdm(dirs)
    for dir in dirs:
        try:
            scan_ftp(ftp, path + [dir], filter[1:], files, depth + 1)
        except Exception as e:
            raise Exception('failed with path `%s`' % ('/'.join(path + [dir]),)) from e


def read_cg67p_img(path):
    pds3_obj_to_band_hack(path)

    img, data, metastr, metadata = read_raw_img(path, (1, 7, 8, 9, 2), gamma=1.8, q_wxyz=False)

    # select only pixel value, model x, y, z and depth
    # - for band indexes, see https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    data[data == 0] = np.nan

    return img, data, metastr, metadata


def pds3_obj_to_band_hack(path):
    pos = 0
    lines = []
    with open(path, 'r') as fh:
        in_obj = False
        extra_space = 0
        substituted = False
        for line in fh:
            line = line[:-1] + '\r\n'  # because carrier return \r in original file disappears
            pos += len(line)
            if not in_obj:
                if re.match(r'^\s*OBJECT\s*=\s*IMAGE\s*$', line):
                    in_obj = True
                lines.append(line)
            else:
                if re.match(r'^\s*END_OBJECT\s*=\s*IMAGE\s*$', line):
                    lines.append(line)
                    break
                elif re.match(r'^\s*BANDS\s*=\s*1\s*$', line):
                    m = re.match(r'^(\s*)BANDS(\s*)=(\s*)1(\s*)$', line)
                    extra_space = sum(map(len, [m[i+1] for i in range(4)])) - 4
                    line1 = ' BANDS = 9\n'
                    line2 = ' BAND_STORAGE_TYPE = BAND_SEQUENTIAL\n'
                    extra_space -= len(line2)
                    if extra_space > 0:
                        line2 = line2[:-1] + (' ' * extra_space) + '\n'
                        extra_space = 0
                    lines.append(line1)
                    lines.append(line2)
                    substituted = True
                else:
                    if extra_space < 0:
                        m = re.match(r'^(\s*\w+\s)(\s*)(\s=\s*\w+\s*)$', line)
                        if m:
                            sp = ' ' * max(0, len(m[2]) + extra_space)
                            line = m[1] + sp + m[3]
                            extra_space += len(m[2]) - len(sp)
                    lines.append(line)

    if not substituted:
        # substitution probably done already
        return

    with open(path, 'rb') as f_in, open(path + '.swp', 'wb') as f_out:
        f_out.write(''.join(lines).encode('ascii'))
        f_in.seek(pos)
        shutil.copyfileobj(f_in, f_out)

    os.unlink(path)
    shutil.move(path + '.swp', path)


if __name__ == '__main__':
    main()

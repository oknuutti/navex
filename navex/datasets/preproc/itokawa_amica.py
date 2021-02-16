import os
import gzip
import shutil
import argparse

from tqdm import tqdm
import numpy as np

from navex.datasets.preproc.tools import read_raw_img, write_data


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
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    for fname in tqdm(os.listdir(args.src)):
        if fname[-4:].lower() == '.lbl':
            path = os.path.join(args.src, fname[:-4])
            extracted = False

            if not os.path.exists(path + '.img'):
                extracted = True
                with gzip.open(path + '.img.gz', 'rb') as fh_in:
                    with open(path + '.img', 'wb') as fh_out:
                        shutil.copyfileobj(fh_in, fh_out)

            img, data = read_itokawa_img(path + '.lbl')
            write_data(os.path.join(args.dst, fname[:-4]), img, data)

            if extracted:
                os.unlink(path + '.img')


def read_itokawa_img(path):
    img, data = read_raw_img(path, (1, 2, 3, 4, 11, 12))

    # select only pixel value and x, y, z; calculate pixel size by taking max of px
    # - for band indexes, see https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/dataset.cat
    px_size = np.atleast_3d(np.max(data[:, :, 3:5], axis=2))
    data = np.concatenate((data[:, :, 0:3], px_size), axis=2)
    data[data <= -1e30] = np.nan

    return img, data


if __name__ == '__main__':
    raw_itokawa()

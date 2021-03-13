import os
import math

import numpy as np
import scipy.interpolate as interp
import cv2

from navex.datasets.base import ImagePairDataset, SynthesizedPairDataset
from navex.datasets.tools import ImageDB, find_files, spherical2cartesian, q_times_v, vector_rejection, angle_between_v, \
    unit_aflow


class AsteroidImagePairDataset(ImagePairDataset):
    def __init__(self, *args, trg_north_ra=None, trg_north_dec=None, **kwargs):
        self.indices, self.index = None, None

        super(AsteroidImagePairDataset, self).__init__(*args, **kwargs)

        self.trg_north_ra, self.trg_north_dec = trg_north_ra, trg_north_dec
        if self.trg_north_ra is None or self.trg_north_dec is None:
            # fall back on ecliptic north, in equatorial ICRF system:
            self.trg_north_ra, self.trg_north_dec = math.radians(270), math.radians(66.56)

    def _load_samples(self):
        dbfile = os.path.join(self.root, 'dataset_all.sqlite')
        self.index = ImageDB(dbfile) if os.path.exists(dbfile) else None

        index = dict(self.index.get_all(('id', 'file')))
        aflow = find_files(os.path.join(self.root, 'aflow'), ext='.png', relative=True)

        get_id = lambda f, i: int(f.split('.')[0].split('_')[i])
        imgs = [(os.path.join(self.root, index[get_id(f, 0)]), os.path.join(self.root, index[get_id(f, 1)]))
                for f in aflow]
        self.indices = [(get_id(f, 0), get_id(f, 1)) for f in aflow]

        samples = list(zip(imgs, aflow))
        return samples

    def preprocess(self, idx, imgs, aflow):

        # TODO: query self.index for relevant params, transform img1, img2 accordingly
        #  (1) Rotate so that image up aligns with up in equatorial (or ecliptic) frame (+z axis)
        #  (2) Rotate so that the sun is directly to the left
        #  (3) Rotate so that asteroid north pole is up
        #  -- thinking by writing:
        #       (3) would probably be the best but don't have the info
        #       (2) would maybe be problematic if close to 0 phase angle as suddenly would maybe need to rotate 180 deg
        #       (1) if know the orientation of the target body rotation axis, could do (3) with only having sc_q!

        # calculate north vector
        north_v = spherical2cartesian(self.trg_north_dec, self.trg_north_ra, 1)

        proc_imgs = []
        for i, img in zip(self.indices[idx], imgs):
            # rotate based on sc_q
            sc_q1 = np.quaternion(*self.index.get(i, ('sc_qw', 'sc_qx', 'sc_qy', 'sc_qz')))
            sc_north = q_times_v(sc_q1.conj(), north_v)

            # project to image plane
            img_north = vector_rejection(sc_north, np.array([1, 0, 0]))

            # calculate angle between projected north vector and image up
            angle = angle_between_v(np.array([0, 0, 1]), img_north)

            # rotate image based on this angle
            img = img.rotate(math.degrees(angle), expand=True, fillcolor=(0, 0, 0))
            proc_imgs.append((np.array([[math.cos(angle), -math.sin(angle)],
                                        [math.sin(angle),  math.cos(angle)]]), img))

        # rotate aflow content so that points to new rotated img2
        oh1, ow1 = aflow.shape[:2]
        r_aflow = aflow.reshape((-1, 2)).dot(proc_imgs[1][0].T).reshape((oh1, ow1, 2))
        min_xy = np.min(aflow, axis=(0, 1))
        aflow = r_aflow - min_xy.reshape((1, 1, 2))

        # rotate aflow indices same way as img1 was rotated
        ifun = interp.RegularGridInterpolator((np.arange(oh1), np.arange(ow1)), aflow, method="nearest",
                                              bounds_error=False, fill_value=np.nan)

        nw1, nh1 = proc_imgs[0][1].size
        grid = unit_aflow(nw1, nh1) - np.array([[[(nw1 - ow1)/2, (nh1 - nw1)/2]]])
        grid = grid.reshape((-1, 2)).dot(np.linalg.inv(proc_imgs[0][0].T)).reshape((nh1, nw1, 2))
        n_aflow = ifun(np.flip(grid, axis=2))

        return [t[1] for t in proc_imgs], n_aflow


class AsteroidSynthesizedPairDataset(SynthesizedPairDataset):
    MIN_FEATURE_INTENSITY = 50

    def valid_area(self, img):
        img = np.array(img)
        if len(img.shape) == 3:
            img = img[:, :, 0]
        _, mask = cv2.threshold(img, self.MIN_FEATURE_INTENSITY, 255, cv2.THRESH_BINARY)
        r = min(*img.shape) // 40
        d = r*2 + 1
        kernel = cv2.circle(np.zeros((d, d), dtype=np.uint8), (r, r), r, 255, -1)
        star_kernel = cv2.circle(np.zeros((9, 9), dtype=np.uint8), (4, 4), 4, 255, -1)

        # exclude asteroid limb from feature detection
        mask = cv2.erode(mask, star_kernel, iterations=1)   # remove stars
        mask = cv2.dilate(mask, kernel, iterations=1)       # remove small shadows inside asteroid
        mask = cv2.erode(mask, kernel, iterations=2)        # remove asteroid limb

        return mask

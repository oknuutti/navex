import os
import math

import numpy as np
import cv2

from navex.datasets.base import ImagePairDataset, SynthesizedPairDataset
from navex.datasets.tools import ImageDB, find_files, spherical2cartesian, q_times_v


class AsteroidImagePairDataset(ImagePairDataset):
    def __init__(self, *args, trg_north_ra=None, trg_north_dec=None, **kwargs):
        self.indices = None
        super(AsteroidImagePairDataset, self).__init__(*args, **kwargs)

        self.trg_north_ra, self.trg_north_dec = trg_north_ra, trg_north_dec
        if self.trg_north_ra is None or self.trg_north_dec is None:
            # fall back on ecliptic north, in equatorial ICRF system:
            self.trg_north_ra, self.trg_north_dec = math.radians(270), math.radians(66.56)

        dbfile = os.path.join(self.root, 'dataset_all.sqlite')
        self.index = ImageDB(dbfile) if os.path.exists(dbfile) else None

    def _load_samples(self):
        index = dict(self.index.get_all(('id', 'file')))
        aflow = find_files(os.path.join(self.root, 'aflow'), ext='.png', relative=True)

        get_id = lambda f, i: int(f.split('.')[0].split('_')[i])
        imgs = [(os.path.join(self.root, index[get_id(f, 0)]), os.path.join(self.root, index[get_id(f, 1)]))
                for f in aflow]
        self.indices = [(get_id(f, 0), get_id(f, 1)) for f in aflow]

        samples = zip(imgs, aflow)
        return samples

    def preprocess(self, idx, imgs, aflow):
        i, j = self.indices[idx]
        img1, img2 = imgs

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

        # rotate based on sc_q
        sc_q1 = np.quaternion(*self.index.get(i, ('sc_qw', 'sc_qx', 'sc_qy', 'sc_qz')))
        north_v = q_times_v(sc_q1.conj(), north_v)

        # project to image plane
        # calculate angle between projected north vector and image up
        # rotate image based on this angle

        return (img1, img2), aflow


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

import os
import math
import sqlite3
import logging

import numpy as np
import quaternion
import scipy.interpolate as interp
import cv2

from navex.datasets.base import ImagePairDataset, SynthesizedPairDataset
from navex.datasets.tools import ImageDB, find_files, spherical2cartesian, q_times_v, vector_rejection, angle_between_v, \
    unit_aflow, show_pair, save_aflow, valid_asteriod_area


class AsteroidImagePairDataset(ImagePairDataset):
    def __init__(self, *args, trg_north_ra=None, trg_north_dec=None, model_north=(0, 0, 1),
                 cam_axis=(0, 0, 1), cam_up=(0, -1, 0), aflow_rot_norm=True, preproc_path=None, **kwargs):
        self.indices, self.index = None, None

        super(AsteroidImagePairDataset, self).__init__(*args, **kwargs)

        self.aflow_rot_norm = aflow_rot_norm
        self.preproc_path = preproc_path
        self.cam_axis, self.cam_up = np.array(cam_axis), np.array(cam_up)
        self.model_north = np.array(model_north)
        self.trg_north_ra, self.trg_north_dec = trg_north_ra, trg_north_dec
        if self.trg_north_ra is None or self.trg_north_dec is None:
            # fall back on ecliptic north, in equatorial ICRF system:
            self.trg_north_ra, self.trg_north_dec = math.radians(270), math.radians(66.56)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['index']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        dbfile = os.path.join(self.root, 'dataset_all.sqlite')
        self.index = ImageDB(dbfile) if os.path.exists(dbfile) else None

    def _load_samples(self):
        dbfile = os.path.join(self.root, 'dataset_all.sqlite')
        self.index = ImageDB(dbfile) if os.path.exists(dbfile) else None

        index = dict(self.index.get_all(('id', 'file')))
        aflow = find_files(os.path.join(self.root, 'aflow'), ext='.png')

        get_id = lambda f, i: int(f.split(os.path.sep)[-1].split('.')[0].split('_')[i])
        aflow = [f for f in aflow if get_id(f, 0) in index and get_id(f, 1) in index]
        self.indices = [(get_id(f, 0), get_id(f, 1)) for f in aflow]
        imgs = [(os.path.join(self.root, index[i]), os.path.join(self.root, index[j])) for i, j in self.indices]

        samples = list(zip(imgs, aflow))
        return samples

    def preprocess(self, idx, imgs, aflow):
        if self.preproc_path is None and not self.aflow_rot_norm:
            return imgs, aflow

        # TODO: query self.index for relevant params, transform img1, img2 accordingly
        #  (1) Rotate so that image up aligns with up in equatorial (or ecliptic) frame (+z axis)
        #  (2) Rotate so that the sun is directly to the left
        #  (3) Rotate so that asteroid north pole is up
        #  -- thinking by writing:
        #       (3) would probably be the best but don't have the info
        #       (2) would maybe be problematic if close to 0 phase angle as suddenly would maybe need to rotate 180 deg
        #       (1) if know the orientation of the target body rotation axis, could do (3) with only having sc_q!

        if not self.aflow_rot_norm:
            # calculate north vector
            north_v = spherical2cartesian(self.trg_north_dec, self.trg_north_ra, 1)
            proc_imgs = []
            for j, (i, img) in enumerate(zip(self.indices[idx], imgs)):
                # rotate based on sc_q and trg_q if both exist, else fallback on sc_q and north_v
                cols = ('sc_qw', 'sc_qx', 'sc_qy', 'sc_qz', 'trg_qw', 'trg_qx', 'trg_qy', 'trg_qz')
                try:
                    q_arr = self.index.get(i, cols)
                except sqlite3.ProgrammingError as e:
                    logging.warning('Got %s, trying again with new db connection' % e)
                    self.index = ImageDB(os.path.join(self.root, 'dataset_all.sqlite'))
                    q_arr = self.index.get(i, cols)
                assert q_arr is not None, 'could not get record id=%s for %s' % (i, self)

                sc_q = quaternion.one if q_arr[0] is None or q_arr[0] == 'None' else np.quaternion(*q_arr[:4])
                trg_q = None if q_arr[4] is None or q_arr[4] == 'None' else np.quaternion(*q_arr[4:])
                if trg_q is None:
                    sc_north = q_times_v(sc_q.conj(), north_v)
                    print('%s: no trg_q!' % self.samples[idx][0][j])
                else:
                    # assuming model frame +z is towards the north pole
                    sc_north = q_times_v(sc_q.conj() * trg_q, self.model_north)

                # project to image plane
                img_north = vector_rejection(sc_north, self.cam_axis)

                # calculate angle between projected north vector and image up  # TODO: correct sign on angle?
                angle = -angle_between_v(self.cam_up, img_north, direction=self.cam_axis)

                # rotate image based on this angle
                img = img.rotate(-math.degrees(angle), expand=True, fillcolor=(0, 0, 0))
                proc_imgs.append((np.array([[math.cos(angle), -math.sin(angle)],
                                            [math.sin(angle),  math.cos(angle)]], dtype=np.float32), img))
        else:
            # calculate rotation angle directly from aflow
            angles = []
            for af, img in zip((unit_aflow(*imgs[0].size), aflow), imgs):
                af = af.reshape((-1, 2))[np.logical_not(np.isnan(aflow[:, :, 0])).flatten(), :]
                v = af - np.mean(af, axis=0)
                angles.append(np.arctan2(v[:, 1], v[:, 0]))
            angle = np.median(((angles[1] - angles[0] + 3*np.pi) % (2*np.pi)) - np.pi)
            img2 = imgs[1].rotate(math.degrees(angle), expand=True, fillcolor=(0, 0, 0))
            proc_imgs = [(np.eye(2, dtype=np.float32), imgs[0]),
                         (np.array([[math.cos(-angle), -math.sin(-angle)],
                                    [math.sin(-angle), math.cos(-angle)]], dtype=np.float32), img2)]

        # rotate aflow content so that points to new rotated img2
        (ow1, oh1), (ow2, oh2) = imgs[0].size, imgs[1].size
        (nw1, nh1), (nw2, nh2) = proc_imgs[0][1].size, proc_imgs[1][1].size
        r_aflow = aflow - np.array([[[ow2/2, oh2/2]]], dtype=np.float32)
        r_aflow = r_aflow.reshape((-1, 2)).dot(proc_imgs[1][0].T).reshape((oh1, ow1, 2))
        r_aflow = r_aflow + np.array([[[nw2/2, nh2/2]]], dtype=np.float32)

        # rotate aflow indices same way as img1 was rotated
        ifun = interp.RegularGridInterpolator((np.arange(-oh1/2, oh1/2, dtype=np.float32),
                                               np.arange(-ow1/2, ow1/2, dtype=np.float32)), r_aflow,
                                              method="nearest", bounds_error=False, fill_value=np.nan)

        grid = unit_aflow(nw1, nh1) - np.array([[[nw1/2, nh1/2]]])
        grid = grid.reshape((-1, 2)).dot(np.linalg.inv(proc_imgs[0][0]).T).reshape((nh1, nw1, 2))
        n_aflow = ifun(np.flip(grid, axis=2).astype(np.float32))

        img1, img2 = [t[1] for t in proc_imgs]
        if 0:
            show_pair(*[t[1] for t in proc_imgs], n_aflow, pts=10, file1=self.samples[idx][0][0],
                                                                   file2=self.samples[idx][0][1])
        if self.aflow_rot_norm:
            return (img1, img2), n_aflow

        (r_img1_pth, r_img2_pth), r_aflow_pth = self.samples[idx]
        folder = getattr(self, 'folder', r_aflow_pth[len(self.root):].strip(os.sep).split(os.sep)[0])
        img1_pth = os.path.join(self.preproc_path, folder, r_img1_pth[len(self.root):].strip(os.sep))
        img2_pth = os.path.join(self.preproc_path, folder, r_img2_pth[len(self.root):].strip(os.sep))
        aflow_pth = os.path.join(self.preproc_path, folder, r_aflow_pth[len(self.root):].strip(os.sep))

        if not os.path.exists(img1_pth):
            os.makedirs(os.path.dirname(img1_pth), exist_ok=True)
            cv2.imwrite(img1_pth, np.array(img1)[:, :, 0], (cv2.IMWRITE_PNG_COMPRESSION, 9))
        if not os.path.exists(img2_pth):
            os.makedirs(os.path.dirname(img2_pth), exist_ok=True)
            cv2.imwrite(img2_pth, np.array(img2)[:, :, 0], (cv2.IMWRITE_PNG_COMPRESSION, 9))
        if not os.path.exists(aflow_pth):
            os.makedirs(os.path.dirname(aflow_pth), exist_ok=True)
            save_aflow(aflow_pth, n_aflow)

        return (img1, img2), n_aflow


class AsteroidSynthesizedPairDataset(SynthesizedPairDataset):
    MIN_FEATURE_INTENSITY = 50

    def valid_area(self, img):
        return valid_asteriod_area(img, min_intensity=self.MIN_FEATURE_INTENSITY)

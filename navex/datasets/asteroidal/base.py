import os
import math
import sqlite3
import logging

import numpy as np
import quaternion
import scipy.interpolate as interp
import cv2

from .. import tools
from ..base import SynthesizedPairDataset, DatabaseImagePairDataset
from ..tools import ImageDB, spherical2cartesian, q_times_v, vector_rejection, angle_between_v, \
    unit_aflow, show_pair, save_aflow, valid_asteriod_area, rotate_expand_border


class AsteroidImagePairDataset(DatabaseImagePairDataset):
    def __init__(self, *args, trg_north_ra=None, trg_north_dec=None, model_north=(0, 0, 1),
                 cam_axis=(0, 0, 1), cam_up=(0, -1, 0), aflow_rot_norm=True, preproc_path=None,
                 extra_crop=None, **kwargs):
        super(AsteroidImagePairDataset, self).__init__(*args, **kwargs)

        # NOTE: impossible to cache aflow rotated images as rotations depend on each pair
        self.skip_preproc = os.path.exists(os.path.join(self.root, 'preprocessed.flag'))
        self.aflow_rot_norm = aflow_rot_norm and not self.skip_preproc
        self.extra_crop = [0, 0, 0, 0] if extra_crop is None else extra_crop   # left, right, top, bottom
        self.preproc_path = preproc_path

        self.cam_axis, self.cam_up = np.array(cam_axis), np.array(cam_up)
        self.model_north = np.array(model_north)
        self.trg_north_ra, self.trg_north_dec = trg_north_ra, trg_north_dec
        if self.trg_north_ra is None or self.trg_north_dec is None:
            # fall back on ecliptic north, in equatorial ICRF system:
            self.trg_north_ra, self.trg_north_dec = math.radians(270), math.radians(66.56)

    def preprocess(self, idx, imgs, aflow):
        assert tuple(np.flip(aflow.shape[:2])) == imgs[0].size, \
            'aflow dimensions do not match with img1 dimensions: %s vs %s' % (np.flip(aflow.shape[:2]), imgs[0].size)

        if self.skip_preproc:
            if 0:
                (r_img1_pth, r_img2_pth), r_aflow_pth = self.samples[idx]
                show_pair(*imgs, aflow, pts=30, file1=r_img1_pth, file2=r_img2_pth, afile=r_aflow_pth)
            return imgs, aflow

        # possibly crop some bad borders
        d = [img.size for img in imgs]
        left, right, top, bottom = self.extra_crop
        right = np.array([d[0][0], d[1][0]]) - right
        bottom = np.array([d[0][1], d[1][1]]) - bottom

        aflow = aflow[top:bottom[0], left:right[0], :] - np.array([[[left, top]]], dtype=aflow.dtype)
        imgs = [img.crop((left, top, right[i], bottom[i])) for i, img in enumerate(imgs)]

        if self.aflow_rot_norm:
            # calculate rotation angle directly from aflow
            angles = []
            for af, img in zip((unit_aflow(*imgs[0].size), aflow), imgs):
                af = af.reshape((-1, 2))[np.logical_not(np.isnan(aflow[:, :, 0])).flatten(), :]
                v = af - np.mean(af, axis=0)
                angles.append(np.arctan2(v[:, 1], v[:, 0]))
            angle = np.median(((angles[1] - angles[0] + 3 * np.pi) % (2 * np.pi)) - np.pi)
            img2 = rotate_expand_border(imgs[1], angle, fullsize=True, lib='opencv', to_pil=True)
            proc_imgs = [(np.eye(2, dtype=np.float32), imgs[0]),
                         (np.array([[math.cos(-angle), -math.sin(-angle)],
                                    [math.sin(-angle), math.cos(-angle)]], dtype=np.float32), img2)]
        else:
            # Query self.index for relevant params, transform img0, img1 so that north is up.
            # First, calculate north vector
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

                # calculate angle between projected north vector and image up
                angle = -angle_between_v(self.cam_up, img_north, direction=self.cam_axis)

                # rotate image based on this angle
                img = rotate_expand_border(img, -angle, fullsize=True, lib='opencv', to_pil=True)
                proc_imgs.append((np.array([[math.cos(angle), -math.sin(angle)],
                                            [math.sin(angle),  math.cos(angle)]], dtype=np.float32), img, angle))

        if 1:
            n_aflow = tools.rotate_aflow(aflow, (imgs[1].size[1], imgs[1].size[0]), proc_imgs[0][2], proc_imgs[1][2],
                                         legacy=True)   # TODO: switch to legacy=False
        else:
            # rotate aflow content so that points to new rotated img1
            (ow1, oh1), (ow2, oh2) = imgs[0].size, imgs[1].size
            (nw1, nh1), (nw2, nh2) = proc_imgs[0][1].size, proc_imgs[1][1].size
            r_aflow = aflow - np.array([[[ow2/2, oh2/2]]], dtype=np.float32)
            r_aflow = r_aflow.reshape((-1, 2)).dot(proc_imgs[1][0].T).reshape((oh1, ow1, 2))
            r_aflow = r_aflow + np.array([[[nw2/2, nh2/2]]], dtype=np.float32)

            # rotate aflow indices same way as img0 was rotated
            ifun = interp.RegularGridInterpolator((np.arange(-oh1/2, oh1/2, dtype=np.float32),
                                                   np.arange(-ow1/2, ow1/2, dtype=np.float32)), r_aflow,
                                                  method="nearest", bounds_error=False, fill_value=np.nan)

            grid = unit_aflow(nw1, nh1) - np.array([[[nw1/2, nh1/2]]])
            grid = grid.reshape((-1, 2)).dot(np.linalg.inv(proc_imgs[0][0]).T).reshape((nh1, nw1, 2))
            n_aflow = ifun(np.flip(grid, axis=2).astype(np.float32))

        img1, img2 = [t[1] for t in proc_imgs]
        if self.preproc_path is None:
            return (img1, img2), n_aflow

        (r_img1_pth, r_img2_pth), r_aflow_pth = self.samples[idx]
        if 0:
            show_pair(img1, img2, n_aflow, pts=20, file1=r_img1_pth, file2=r_img2_pth, afile=r_aflow_pth)
        folder = getattr(self, 'folder', r_aflow_pth[len(self.root):].strip(os.sep).split(os.sep)[0])

        img1_pth = os.path.join(self.preproc_path, folder, r_img1_pth[len(self.root):].strip(os.sep))
        os.makedirs(os.path.dirname(img1_pth), exist_ok=True)
        cv2.imwrite(img1_pth, np.array(img1)[:, :, 0], (cv2.IMWRITE_PNG_COMPRESSION, 9))

        img2_pth = os.path.join(self.preproc_path, folder, r_img2_pth[len(self.root):].strip(os.sep))
        os.makedirs(os.path.dirname(img2_pth), exist_ok=True)
        cv2.imwrite(img2_pth, np.array(img2)[:, :, 0], (cv2.IMWRITE_PNG_COMPRESSION, 9))

        aflow_pth = os.path.join(self.preproc_path, folder, r_aflow_pth[len(self.root):].strip(os.sep))
        os.makedirs(os.path.dirname(aflow_pth), exist_ok=True)
        save_aflow(aflow_pth, n_aflow)

        return (img1, img2), n_aflow


class AsteroidSynthesizedPairDataset(SynthesizedPairDataset):
    MIN_FEATURE_INTENSITY = 50

    def valid_area(self, img):
        return valid_asteriod_area(img, min_intensity=self.MIN_FEATURE_INTENSITY)

import itertools
import math
import os
import sqlite3
import logging

import numpy as np
import cv2
import PIL

from ..base import SynthesizedPairDataset, DatabaseImagePairDataset, AugmentedPairDatasetMixin
from ..tools import find_files_recurse, resize_aflow, ImageDB, show_pair, save_aflow


class BatvikSynthPairDataset(SynthesizedPairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='batvik', subset=None, max_tr=0, max_rot=math.radians(15), max_shear=0.2,
                 max_proj=0.8, noise_max=0.20, rnd_gain=(0.5, 2), image_size=512, max_sc=2**(1/4), margin=16,
                 eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        self.folder = folder
        self.subset = subset

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=max_sc, margin=margin, eval=eval, rgb=rgb, blind_crop=True)

        SynthesizedPairDataset.__init__(self, os.path.join(root, self.folder), max_tr=max_tr,
                                        max_rot=max_rot, max_shear=max_shear, max_proj=max_proj,
                                        min_size=image_size // 2)

    def _load_samples(self):
        paths = [self.root]
        if self.subset is not None:
            paths = [os.path.join(self.root, s) for s in self.subset]
        return list(itertools.chain(*[find_files_recurse(path, ext='.jpg') for path in paths]))


class BatvikPairDataset(DatabaseImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='batvik', noise_max=0.0, rnd_gain=1.0, image_size=512,
                 margin=16, eval=False, rgb=False, npy=False, resize_max_sc=1.0, fixed_ground_res=None,
                 preproc_path=None):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'
        self.folder = folder
        self.fixed_ground_res = fixed_ground_res
        self.preproc_path = preproc_path

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           max_sc=1.0, margin=margin, fill_value=0, eval=eval, rgb=False,
                                           resize_max_sc=resize_max_sc, blind_crop=False)
        DatabaseImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms)

    def preprocess(self, idx, imgs, aflow):
        assert tuple(np.flip(aflow.shape[:2])) == imgs[0].size, \
            'aflow dimensions do not match with img1 dimensions: %s vs %s' % (np.flip(aflow.shape[:2]), imgs[0].size)

        if self.fixed_ground_res is None:
            if 0:
                (r_img1_pth, r_img2_pth), r_aflow_pth = self.samples[idx]
                show_pair(*imgs, aflow, pts=30, file1=r_img1_pth, file2=r_img2_pth, afile=r_aflow_pth)
            return imgs, aflow

        # Query self.index for relevant params, resize img0, img1 so that ground resolution is correct
        proc_imgs = []
        for j, (i, img) in enumerate(zip(self.indices[idx], imgs)):
            # rotate based on sc_q and trg_q if both exist, else fallback on sc_q and north_v
            cols = ('sc_trg_z', 'hz_fov')   # assume nadir pointing, ground `sc_trg_z` meters away
            try:
                q_arr = self.index.get(i, cols)
            except sqlite3.ProgrammingError as e:
                logging.warning('Got %s, trying again with new db connection' % e)
                self.index = ImageDB(os.path.join(self.root, 'dataset_all.sqlite'))
                q_arr = self.index.get(i, cols)
            assert q_arr is not None, 'could not get record id=%s for %s' % (i, self)

            # calc sc
            w, h = img.size
            alt, hz_fov = q_arr
            fl_x = (w/2)/math.tan(math.radians(hz_fov/2))
            ground_res = alt / fl_x
            sc = ground_res / self.fixed_ground_res

            # resize image based on this scale
            img = cv2.resize(np.array(img), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            img = PIL.Image.fromarray(img)
            proc_imgs.append((img, sc))

        img1, img2 = [t[0] for t in proc_imgs]
        sc2 = proc_imgs[1][1]
        n_aflow = resize_aflow(aflow, img1.size, img2.size, sc2)

        (r_img1_pth, r_img2_pth), r_aflow_pth = self.samples[idx]
        if 1:
            show_pair(img1, img2, n_aflow, pts=20, file1=r_img1_pth, file2=r_img2_pth, afile=r_aflow_pth)

        if self.preproc_path is None:
            return (img1, img2), n_aflow

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

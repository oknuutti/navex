import math
import argparse
import logging
import os
from typing import Union

from tqdm import tqdm
import numpy as np
import quaternion
import cv2

import torch
from torch.utils.data import DataLoader

from navex.datasets.aerial.batvik import BatvikPairDataset
from navex.datasets.transforms import PairCenterCrop
from navex.models.tools import MatchException
from navex.visualizations.misc import tensor2img, img2tensor

try:
    import featstat.algo.model as fsm
    from featstat.algo.odo.simple import SimpleOdometry
    from featstat.algo import tools as fs_tools
except ImportError:
    fsm = SimpleOdometry = fs_tools = None

from .datasets.asteroidal.cg67p import CG67pOsinacPairDataset
from .datasets.asteroidal.eros import ErosPairDataset
from .datasets.asteroidal.itokawa import ItokawaPairDataset
from .datasets.asteroidal.synth import SynthBennuPairDataset
from .datasets.tools import Camera, q_to_ypr, from_opencv_q, load_mono, estimate_pose_pnp, \
    nan_grid_interp
from .datasets import tools as ds_tools
from .models import tools
from .extract import Extractor

HEADER = ['Dataset', 'aflow', 'img1', 'img2', 'light1_x', 'light1_y', 'light1_z', 'light2_x', 'light2_y', 'light2_z',
          'rel_qw', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_angle', 'rel_dist',  'FD', 'M-Score', 'MMA', 'LE', 'mAP',
          'dist-err', 'lat-err', 'ori-err', 'est_qw', 'est_qx', 'est_qy', 'est_qz', 'dist']

logger = fs_tools.get_logger("main", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser("evaluate a feature extractor")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--output", type=str, required=True, help='csv file for evaluation output')
    parser.add_argument("--root", type=str, default='data', help='root folder of all datasets')
    parser.add_argument("--dataset", "-d", choices=('eros', 'ito', '67p', 'synth', 'batvik'), action='append',
                        help='Selected dataset, can give multiple, default: all except batvik')
    parser.add_argument("--top-k", type=int, default=None, help='Limit on total number of keypoints')
    parser.add_argument("--feat-d", type=float, default=0.001, help='Number of keypoints per pixel')
    parser.add_argument("--scale-f", type=float, default=2 ** (1/4))
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--crop", type=int, default=1, help='Crop images to max-size instead of down-scaling')
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--det-lim", type=float, default=0.7)
    parser.add_argument("--qlt-lim", type=float, default=0.7)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--det-mode", default='nms')
    parser.add_argument("--border", type=int, default=16, help="Dont detect features if this close to image border")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--mutual", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--success-px-limit", "--px-lim", type=float, default=5.0)
    parser.add_argument("--est-pose", action="store_true", help="Estimate image pair pose change based on "
                                                                "matched features, not properly implemented yet")
    parser.add_argument("--est-gt-pose", default=1, type=int, help="Recalculate ground truth relative pose "
                                                                   "based on aflow. Default: 1, due to the values "
                                                                   "available in the database being incorrect.")
    parser.add_argument("--ignore-img-rot", action="store_true", help="Assume images have not been rotated, useful "
                                                                      "when dataset_all.sqlite has non-zero img_angle "
                                                                      "even though images are untouched")
    parser.add_argument("--rotated-depthmaps", action="store_true", help="The depthmaps have already been rotated in "
                                                                         "the dataset, so don't rotate them again")
    parser.add_argument("--show-matches", action="store_true", help="Show matches if estimating pose")
    parser.add_argument("--debug-pose-est", action="store_true", help="Debug pose estimation")
    args = parser.parse_args()

    ext = Extractor(args.model, gpu=args.gpu, top_k=args.top_k, feat_d=args.feat_d, border=args.border,
                    scale_f=args.scale_f, min_size=args.min_size, max_size=args.max_size,
                    min_scale=args.min_scale, max_scale=args.max_scale, det_lim=args.det_lim,
                    qlt_lim=args.qlt_lim, mode=args.det_mode, kernel_size=args.kernel_size)

    pose_est = None
    if args.est_pose:
        assert SimpleOdometry is not None, "featstat package not installed, cannot estimate pose without it"
        pose_est = PoseEstimator(max_repr_err=args.success_px_limit, min_inliers=12, use_ba=True,
                                 debug=args.debug_pose_est, show_matches=args.show_matches,
                                 rotated_depthmaps=args.rotated_depthmaps)

    eval = ImagePairEvaluator(ext, pose_est, args.success_px_limit, args.mutual, args.ratio,
                              est_gt_pose=args.est_gt_pose)

    if 'batvik' in args.dataset:
        datasets = {'batvik': BatvikPairDataset}
    else:
        datasets = {'eros': ErosPairDataset, '67p': CG67pOsinacPairDataset, 'ito': ItokawaPairDataset,
                    'synth': SynthBennuPairDataset}
        datasets = {key: datasets[key] for key in args.dataset} if args.dataset else datasets

    write_header(args)

    for key, DatasetClass in datasets.items():
        dataset = DatasetClass(root=args.root, eval='test', image_size=args.max_size if args.crop else None)
        if args.crop:
            tfs = dataset.transforms.transforms
            tfs[[t.__class__ for t in tfs].index(PairCenterCrop)].return_bounds = True
            # dataset.cam.resolution = (args.max_size, args.max_size)
            # dataset.cam.matrix[0:2, 2] = args.max_size / 2
        dataset.skip_preproc = True if args.ignore_img_rot else dataset.skip_preproc

        sampler = None
        if 0:
            ids = [s[1].split(os.sep)[-1].split('.')[0] for s in dataset.samples]
            sampler = [ids.index('553_23454')]

        pbar = tqdm(DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, sampler=sampler),
                    desc="Evaluating %s..." % key)
        for i, ((img1, img2), aflow, *meta) in enumerate(pbar):
            assert len(meta) >= 4, 'dataset "%s" does not have ids, rel_dist, img_rot1, img_rot2 metadata' % (key,)

            # as in datasets/base.py: DatabaseImagePairDataset._load_samples
            (id1, id2), rel_dist, img_rot1, img_rot2, cf_trg_q1, cf_trg_q2, light1, light2, *xtra = meta

            if hasattr(dataset, 'cam'):
                cam1 = cam2 = dataset.cam
            elif hasattr(dataset, 'get_cam'):
                cam1, cam2 = map(dataset.get_cam, (id1, id2))
            else:
                assert False, 'dataset "%s" does not have cam model' % (key,)

            if args.ignore_img_rot:
                img_rot1 = img_rot2 = 0.0

            cf_trg_q1 = np.quaternion(*cf_trg_q1.flatten().tolist())
            cf_trg_q2 = np.quaternion(*cf_trg_q2.flatten().tolist())
            cf_trg_rel_q = cf_trg_q1.conj() * cf_trg_q2

            tf_cam_axis = np.array([-1, 0, 0])
            tf_cam_q1, tf_cam_q2 = cf_trg_q1.conj(), cf_trg_q2.conj()
            rel_angle = math.degrees(ds_tools.angle_between_v(ds_tools.q_times_v(tf_cam_q1, tf_cam_axis),
                                                              ds_tools.q_times_v(tf_cam_q2, tf_cam_axis)))

            depth1 = None
            if args.est_pose:
                path = dataset.samples[i][0][0][:-4] + '.d'
                depth1 = load_mono(path if os.path.exists(path) else path + '.exr')

            try:
                metrics = eval.evaluate(img1, img2, aflow, img_rot1, img_rot2, cam1, cam2, depth1, cf_trg_rel_q,
                                        crop_bounds=xtra[0] if args.crop else None)
            except EvaluationException as e:
                fa, (f1, f2) = dataset.samples[i][1], dataset.samples[i][0]
                logger.warning(f'During evaluation of {fa}, {f1}, {f2}:\n\t{e}')
                metrics = e.result
            write_row(args.output, key, dataset.samples[i][1], *dataset.samples[i][0],
                      light1, light2, cf_trg_rel_q, rel_angle, rel_dist, metrics)


def write_header(args):
    anames = [arg for arg in dir(args) if not arg.startswith('_') and arg not in ('output',)]
    avals = [getattr(args, name) for name in anames]
    header = HEADER if args.est_pose else HEADER[:-8]

    with open(args.output, 'w') as fh:
        fh.write('\t'.join(anames) + '\n')
        fh.write('\t'.join(map(str, avals)) + '\n\n')
        fh.write('\t'.join(header) + '\n')


def write_row(file, ds, aflow, img1, img2, light1, light2, cf_trg_rel_q, rel_angle, rel_dist, metrics):
    root = os.path.commonpath([aflow, img1, img2])
    rlen = len(root) + 1

    with open(file, 'a') as fh:
        fh.write('\t'.join(map(str, (ds, aflow[rlen:], img1[rlen:], img2[rlen:],
                                     *light1.flatten().tolist(), *light2.flatten().tolist(),
                                     *cf_trg_rel_q.components, rel_angle, rel_dist.item(), *metrics))) + '\n')


class EvaluationException(Exception):
    def __init__(self, msg, result=None):
        super().__init__(msg)
        self.result = result if result is not None else 2 * [0.] + 11 * [float('nan')]


class ImagePairEvaluator:
    def __init__(self, extractor: Extractor, pose_est: 'PoseEstimator', success_px_limit, mutual, ratio,
                 est_gt_pose=True):
        self.extractor = extractor
        self.pose_est = pose_est
        self.success_px_limit = success_px_limit
        self.mutual = mutual
        self.ratio = ratio
        self.est_gt_pose = est_gt_pose

    def evaluate(self, img1, img2, aflow, img_rot1, img_rot2, cam1, cam2, depth1, rel_q, crop_bounds=None):
        if depth1.shape != (cam1.height, cam1.width):
            raise EvaluationException(f"Unexpected image size: ({depth1.shape[0]}, {depth1.shape[1]})")

        xys1, desc1, scores1 = self.extractor.extract(img1)
        xys2, desc2, scores2 = self.extractor.extract(img2)
        syx1 = torch.flipud(torch.tensor(xys1.T))[None, :, :]    # [K1, XYS] => [1, SYX, K1]
        syx2 = torch.flipud(torch.tensor(xys2.T))[None, :, :]
        desc1 = torch.tensor(desc1.T)[None, :, :]     # [K1, D]   => [1, D, K1]
        desc2 = torch.tensor(desc2.T)[None, :, :]

        _, _, H1, W1 = img1.shape
        _, _, H2, W2 = img2.shape
        norm = 'hamming' if desc1.dtype == torch.uint8 else 2

        if 1:
            try:
                # [B, K1], [B, K1], [B, K1], [B, K1, K2], [B, K1], [B, K2]
                matches, mdist, mask, dist, m1, m2 = tools.scale_restricted_match(syx1, desc1, syx2, desc2, norm=norm,
                                                                                  mutual=self.mutual, ratio=self.ratio,
                                                                                  type='windowed')
            except MatchException as e:
                raise EvaluationException(f'Matching failed due to: {e}')

            assert matches.shape[0] == 1, 'batch size > 1 not supported'
            matches, mask, dist = matches[:1, m1[0]], mask[:1, m1[0]], dist[:1, m1[0], :][:, :, m2[0]]
            yx1, yx2 = syx1[:, 1:, m1[0]].type(torch.long), syx2[:, 1:, m2[0]].type(torch.long)
            xys1, xys2 = xys1[m1[0], :], xys2[m2[0], :]
        else:
            # [B, K1], [B, K1], [B, K1], [B, K1, K2]
            matches, mdist, mask, dist = tools.match(desc1, desc2, norm=norm, mutual=self.mutual, ratio=self.ratio)
            yx1, yx2 = syx1[:, 1:, :].type(torch.long), syx2[:, 1:, :].type(torch.long)

        if self.pose_est.show_matches and 1:
            PoseEstimator._plot_matches(img1, xys1[:, :2], img2, xys2[:, :2], aflow, matches, mask,
                                        title=f'all matches (n={mask.sum()})')

        metrics = tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), self.success_px_limit,
                                      border=self.extractor.border)
        metrics = metrics.flatten().tolist()

        # calc relative pose error
        if self.pose_est is not None:
            if self.est_gt_pose:
                rel_p, rel_q = self.pose_est.estimate_gt(aflow, depth1, img_rot1, (W1, H1), cam1,
                                                         img_rot2, (W2, H2), cam2,
                                                         max_repr_err=0.75, min_inliers=30, crop_bounds=crop_bounds,
                                                         debug=(img1, img2, aflow, rel_q))
                median_dist = self.pose_est.median_dist

            if rel_q is not None:
                est_p, est_q = self.pose_est.estimate(yx1, yx2, matches, mask, depth1, img_rot1, (W1, H1), cam1,
                                                      img_rot2, (W2, H2), cam2,
                                                      crop_bounds=crop_bounds,
                                                      debug=(img1, img2, aflow, rel_q))

            if rel_q is not None and est_q is not None:
                ori_err = math.degrees(ds_tools.angle_between_q(est_q, rel_q))
                dist_err = (est_p[0] - rel_p[0]) / median_dist
                lat_err = np.linalg.norm(est_p[1:] - rel_p[1:]) / median_dist
                metrics = metrics + [dist_err, lat_err, ori_err, *est_q.components, median_dist]
            elif rel_q is None:
                raise EvaluationException('Failed to estimate ground truth relative orientation',
                                          metrics + [np.nan] * 8)
            else:
                metrics = metrics + [np.inf] * 3 + [np.nan] * 5

        return metrics


class PoseEstimator:
    def __init__(self, max_repr_err=1.5, min_inliers=25, use_ba=True, rotated_depthmaps=True,
                 debug=False, show_matches=False):
        self.max_repr_err = max_repr_err
        self.min_inliers = min_inliers
        self.use_ba = use_ba
        self.rotated_depthmaps = rotated_depthmaps
        self.debug = debug
        self.show_matches = show_matches
        self._debug_logger = fs_tools.get_logger("odo", level=logging.DEBUG) if self.debug > 1 else None
        self.median_dist = None

    def estimate_gt(self, aflow, depth1, img_rot1, size1, cam1: Camera, img_rot2, size2, cam2: Camera, crop_bounds=None,
                    debug=None, **kwargs):
        kwargs.update({'debug': False, 'show_matches': False})

        tmp = self.__dict__.copy()
        for k, v in kwargs.items():
            setattr(self, k, v)

        unitflow = ds_tools.unit_aflow(aflow.shape[3], aflow.shape[2])
        xy1 = unitflow[torch.logical_not(torch.isnan(aflow[0, 0, :, :])).cpu().numpy()].reshape((-1, 2))

        # sparsify xy1
        if len(xy1) > 20000:
            n = len(xy1) // 10000
            xy1 = xy1[::n, :]

        xy2 = aflow[0, :, xy1[:, 1].astype(int), xy1[:, 0].astype(int)].cpu().numpy().T
        mask = np.logical_not(np.isnan(xy2[:, 0]))
        matches = np.arange(len(mask))

        gt_p, gt_q = self.estimate(xy1, xy2, matches, mask, depth1, img_rot1, size1, cam1, img_rot2, size2, cam2,
                                   crop_bounds=crop_bounds, debug=debug)

        for k in kwargs.keys():
            setattr(self, k, tmp[k])

        return gt_p, gt_q

    def estimate(self, yx1, yx2, matches, mask, depth1, img_rot1, size1, cam1: Camera, img_rot2, size2, cam2: Camera,
                 crop_bounds=None, debug=None):
        if isinstance(yx1, np.ndarray):
            xy1, xy2 = yx1, yx2
        else:
            xy1, xy2 = map(lambda yx: torch.flipud(yx[0, :, :]).t().cpu().numpy(), (yx1, yx2))
            matches, mask = map(lambda m: m[0, :].cpu().numpy(), (matches, mask))

        if np.sum(mask) < self.min_inliers:
            return None, None

        if debug is not None and (self.debug or self.show_matches):
            img1, img2, aflow, rel_q = debug

        if crop_bounds is not None:
            (i1s, j1s, i1e, j1e, w1, h1) = map(lambda x: x.item(), crop_bounds[0])
            (i2s, j2s, i2e, j2e, w2, h2) = map(lambda x: x.item(), crop_bounds[1])
            size1, size2 = (w1, h1), (w2, h2)
            xy1[:, 0] += i1s
            xy1[:, 1] += j1s
            xy2[:, 0] += i2s
            xy2[:, 1] += j2s

        if self.rotated_depthmaps:
            dist = nan_grid_interp(depth1, xy1[mask, :], max_radius=self.max_repr_err)

        # rotate keypoints back to original image coordinates
        xy1 = self.rotate_kps(xy1, -img_rot1, size1, (cam1.width, cam1.height))
        xy2 = self.rotate_kps(xy2, -img_rot2, size2, (cam2.width, cam2.height))

        # mark keypoints outside of images as invalid
        mask = np.logical_and(mask, np.logical_and.reduce((xy1[:, 0] >= -0.5, xy1[:, 1] >= -0.5,
                                                           xy1[:, 0] < cam1.width-0.5, xy1[:, 1] < cam1.height-0.5)))
        mxy2 = xy2[matches[mask], :]
        mask[mask] = np.logical_and.reduce((mxy2[:, 0] >= -0.5, mxy2[:, 1] >= -0.5, mxy2[:, 0] < cam2.width-0.5,
                                            mxy2[:, 1] < cam2.height-0.5))

        if not self.rotated_depthmaps:
            dist = nan_grid_interp(depth1, xy1[mask, :], max_radius=self.max_repr_err)

        self.median_dist = np.nanmedian(dist)
        kp3d = cam1.backproject(xy1[mask, :], dist=dist)
        repr_err_callback = (lambda kps, errs: self._plot_repr_errs(img2, kps, errs)) if self.debug else None
        pos, est_q, inliers = estimate_pose_pnp(cam2, kp3d, xy2[matches[mask], :].astype(float),
                                                ransac=not self.use_ba, ba=self.use_ba,
                                                max_err=self.max_repr_err, input_in_opencv=True,
                                                repr_err_callback=repr_err_callback)
        if est_q is None or len(inliers) < self.min_inliers:
            return None, None

        if debug is not None and (self.debug or self.show_matches):
            ry, rp, rr = map(math.degrees, q_to_ypr(rel_q))
            ey, ep, er = map(math.degrees, q_to_ypr(est_q))
            logger.info('rel ypr: %.1f, %.1f, %.1f' % (ry, rp, rr))
            logger.info('est ypr: %.1f, %.1f, %.1f' % (ey, ep, er))

            if self.show_matches and 1:
                nsize = (cam2.width, cam2.height) if crop_bounds is None else 'full'
                np_img1, np_img2 = map(tensor2img, (img1, img2))
                np_img1 = ds_tools.rotate_array(np_img1, -img_rot1, new_size=nsize, border=cv2.BORDER_CONSTANT)
                np_img2 = ds_tools.rotate_array(np_img2, -img_rot2, new_size=nsize, border=cv2.BORDER_CONSTANT)
                np_aflow = aflow[0, ...].permute((1, 2, 0)).cpu().numpy()
                np_aflow = ds_tools.rotate_aflow(np_aflow, (img2.shape[3], img2.shape[2]), -img_rot1, -img_rot2,
                                                 new_size1=nsize, new_size2=nsize)
                aflow = torch.Tensor(np_aflow[None, ...]).permute((0, 3, 1, 2))
                img1, img2 = map(img2tensor, (np_img1, np_img2))

                xoff1, yoff1 = (0, 0) if crop_bounds is None else \
                    self.rotated_crop_offset((i1s, j1s, i1e, j1e), -img_rot1, size1, (cam1.width, cam1.height))
                xoff2, yoff2 = (0, 0) if crop_bounds is None else \
                    self.rotated_crop_offset((i2s, j2s, i2e, j2e), -img_rot2, size2, (cam2.width, cam2.height))

                res_mask = np.zeros_like(mask)
                res_mask[np.where(mask)[0][inliers]] = True
                self._plot_matches(img1, xy1 - np.array([[xoff1, yoff1]]),
                                   img2, xy2 - np.array([[xoff2, yoff2]]),
                                   aflow, matches, res_mask, title=f'inliers (n={inliers.size})', kps_only=True)

        return pos, est_q

    @staticmethod
    def rotated_crop_offset(crop_bounds, img_rot, size0, size1):
        i0, j0, i1, j1 = crop_bounds
        pts = np.array([[i0, j0], [i1, j0], [i1, j1], [i0, j1]])
        pts = PoseEstimator.rotate_kps(pts, img_rot, size0, size1)
        return pts.min(axis=0).astype(int)

    @staticmethod
    def rotate_kps(xy, angle, size0, size1):
        R = np.array([[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
        rxy = (xy - np.array(size0).reshape((1, 2))/2).dot(R) + np.array(size1).reshape((1, 2))/2
        return rxy

    @staticmethod
    def _plot_repr_errs(img, xy, errs):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, squeeze=False)
        if 0:
            from .visualizations.misc import tensor2img
            img = cv2.cvtColor(tensor2img(img), cv2.COLOR_GRAY2RGB)
            axs[0, 0].imshow(img)
        axs[0, 0].scatter(xy[:, 0], xy[:, 1], np.clip(errs*30, 1, 10), facecolors='none', edgecolors='C0')
        axs[0, 0].set_aspect('equal')
        axs[0, 0].invert_yaxis()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_matches(img1: torch.Tensor, xy1: np.ndarray, img2: torch.Tensor, xy2: np.ndarray, aflow: torch.Tensor,
                      matches: Union[torch.Tensor, np.ndarray], mask: Union[torch.Tensor, np.ndarray],
                      title=None, kps_only=False):
        matches = matches[0, :].cpu().numpy() if isinstance(matches, torch.Tensor) else matches
        mask = mask[0, :].cpu().numpy().astype(bool) if isinstance(mask, torch.Tensor) else mask.astype(bool)
        img = PoseEstimator._draw_matches(img1, xy1[mask, :], img2, xy2[matches[mask], :], kps_only=kps_only)

        gt_xy2 = aflow[0, :, (xy1[mask, 1] + 0.5).astype(int), (xy1[mask, 0] + 0.5).astype(int)].cpu().numpy().T
        gt_mask1, gt_mask2 = mask.copy(), ~np.isnan(gt_xy2[:, 0])
        gt_mask1[mask] = gt_mask2
        img_gt = PoseEstimator._draw_matches(img1, xy1[gt_mask1, :], img2, gt_xy2[gt_mask2, :])

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        axs[0].imshow(img)
        axs[1].imshow(img_gt)
        if title is not None:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _draw_matches(img1, xy1, img2, xy2, kps_only=False):
        from .visualizations.misc import tensor2img
        img1, img2 = map(lambda img: cv2.cvtColor(tensor2img(img), cv2.COLOR_GRAY2RGB), (img1, img2))
        kp1, kp2 = map(lambda xy: [cv2.KeyPoint(x, y, 1) for x, y in xy.astype(float)], (xy1, xy2))
        if kps_only:
            img = np.zeros_like(img1, shape=(max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3))
            cv2.drawKeypoints(img1, kp1, img[:img1.shape[0], :img1.shape[1], :], color=(0, 0, 255))
            cv2.drawKeypoints(img2, kp2, img[:img2.shape[0], img1.shape[1]:, :], color=(0, 0, 255))
        else:
            matches = [cv2.DMatch(i, i, np.random.uniform(1, 2)) for i in range(len(kp1))]
            img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, singlePointColor=(0, 0, 255))
        return img


if __name__ == '__main__':
    main()

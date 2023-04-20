import math
import argparse
import logging
import os

from tqdm import tqdm
import numpy as np
import quaternion
import cv2

import torch
from torch.utils.data import DataLoader

from navex.models.tools import MatchException

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
          'ori-err', 'est_qw', 'est_qx', 'est_qy', 'est_qz']

logger = fs_tools.get_logger("main", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser("evaluate a feature extractor")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--output", type=str, required=True, help='csv file for evaluation output')
    parser.add_argument("--root", type=str, default='data', help='root folder of all datasets')
    parser.add_argument("--dataset", "-d", choices=('eros', 'ito', '67p', 'synth'), action='append',
                        help='selected dataset, can give multiple, default: all')
    parser.add_argument("--top-k", type=int, default=None, help='limit on total number of keypoints')
    parser.add_argument("--feat-d", type=float, default=0.001, help='number of keypoints per pixel')
    parser.add_argument("--scale-f", type=float, default=2 ** (1/4))
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--det-lim", type=float, default=0.7)
    parser.add_argument("--qlt-lim", type=float, default=0.7)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--det-mode", default='nms')
    parser.add_argument("--border", type=int, default=16, help="dont detect features if this close to image border")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--mutual", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--success-px-limit", "--px-lim", type=float, default=5.0)
    parser.add_argument("--est-ori", action="store_true", help="Estimate image pair orientation change based on "
                                                               "matched features, not properly implemented yet")
    parser.add_argument("--est-gt-ori", default=1, type=int, help="Recalculate ground truth relative orientation "
                                                                  "based on aflow. Default: 1, due to the values "
                                                                  "available in the database being incorrect.")
    parser.add_argument("--ignore-img-rot", action="store_true", help="Assume images have not been rotated, useful "
                                                                      "when dataset_all.sqlite has non-zero img_angle "
                                                                      "even though images are untouched")
    parser.add_argument("--show-matches", action="store_true", help="Show matches if estimating orientation")
    parser.add_argument("--debug-ori-est", action="store_true", help="Debug orientation estimation")
    args = parser.parse_args()

    ext = Extractor(args.model, gpu=args.gpu, top_k=args.top_k, feat_d=args.feat_d, border=args.border,
                    scale_f=args.scale_f, min_size=args.min_size, max_size=args.max_size,
                    min_scale=args.min_scale, max_scale=args.max_scale, det_lim=args.det_lim,
                    qlt_lim=args.qlt_lim, mode=args.det_mode, kernel_size=args.kernel_size)

    ori_est = None
    if args.est_ori:
        assert SimpleOdometry is not None, "featstat package not installed, cannot estimate orientation without it"
        ori_est = OrientationEstimator(max_repr_err=args.success_px_limit, min_inliers=12, use_ba=True,
                                       debug=args.debug_ori_est, show_matches=args.show_matches)

    eval = ImagePairEvaluator(ext, ori_est, args.success_px_limit, args.mutual, args.ratio,
                              est_gt_ori=args.est_gt_ori)

    datasets = {'eros': ErosPairDataset, '67p': CG67pOsinacPairDataset, 'ito': ItokawaPairDataset,
                'synth': SynthBennuPairDataset}
    datasets = {key: datasets[key] for key in args.dataset} if args.dataset else datasets

    write_header(args)

    for key, DatasetClass in datasets.items():
        dataset = DatasetClass(root=args.root, eval='test')
        dataset.skip_preproc = True if args.ignore_img_rot else dataset.skip_preproc

        sampler = None
        if 0:
            ids = [s[1].split(os.sep)[-1].split('.')[0] for s in dataset.samples]
            sampler = [ids.index('553_23454')]

        pbar = tqdm(DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, sampler=sampler),
                    desc="Evaluating %s..." % key)
        for i, ((img1, img2), aflow, *meta) in enumerate(pbar):
            assert len(meta) >= 2 and dataset.cam, 'dataset "%s" does not have cam model or rel_q, rel_s metadata' % (key,)

            # as in datasets/base.py: DatabaseImagePairDataset._load_samples
            rel_dist, img_rot1, img_rot2, cf_trg_q1, cf_trg_q2, light1, light2 = meta

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
            if args.est_ori:
                path = dataset.samples[i][0][0][:-4] + '.d'
                depth1 = load_mono(path if os.path.exists(path) else path + '.exr')

            try:
                metrics = eval.evaluate(img1, img2, aflow, img_rot1, img_rot2, dataset.cam, depth1, cf_trg_rel_q)
            except EvaluationException as e:
                fa, (f1, f2) = dataset.samples[i][1], dataset.samples[i][0]
                logger.warning(f'During evaluation of {fa}, {f1}, {f2}: {e}')
                metrics = e.result
            write_row(args.output, key, dataset.samples[i][1], *dataset.samples[i][0],
                      light1, light2, cf_trg_rel_q, rel_angle, rel_dist, metrics)


def write_header(args):
    anames = [arg for arg in dir(args) if not arg.startswith('_') and arg not in ('output',)]
    avals = [getattr(args, name) for name in anames]
    header = HEADER if args.est_ori else HEADER[:-5]

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
        self.result = result if result is not None else 2 * [0.] + 8 * [float('nan')]


class ImagePairEvaluator:
    def __init__(self, extractor: Extractor, ori_est: 'OrientationEstimator', success_px_limit, mutual, ratio,
                 est_gt_ori=True):
        self.extractor = extractor
        self.ori_est = ori_est
        self.success_px_limit = success_px_limit
        self.mutual = mutual
        self.ratio = ratio
        self.est_gt_ori = est_gt_ori

    def evaluate(self, img1, img2, aflow, img_rot1, img_rot2, cam, depth1, rel_q):
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
                                                                                  type='hires')
            except MatchException as e:
                raise EvaluationException(f'Matching failed due to: {e}')

            assert matches.shape[0] == 1, 'batch size > 1 not supported'
            matches, mask, dist = matches[:1, m1[0]], mask[:1, m1[0]], dist[:1, m1[0], :][:, :, m2[0]]
            yx1, yx2 = syx1[:, 1:, m1[0]].type(torch.long), syx2[:, 1:, m2[0]].type(torch.long)
        else:
            # [B, K1], [B, K1], [B, K1], [B, K1, K2]
            matches, mdist, mask, dist = tools.match(desc1, desc2, norm=norm, mutual=self.mutual, ratio=self.ratio)
            yx1, yx2 = syx1[:, 1:, :].type(torch.long), syx2[:, 1:, :].type(torch.long)

        brd2 = self.extractor.border * 2
        metrics = tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), self.success_px_limit,
                                      active_area=((H1 - brd2) * (W1 - brd2) + (H2 - brd2) * (W2 - brd2)) / 2)
        metrics = metrics.flatten().tolist()

        # calc relative orientation error
        if self.ori_est is not None:
            if self.est_gt_ori:
                rel_q = self.ori_est.estimate_gt(aflow, depth1, img_rot1, (W1, H1), img_rot2, (W2, H2), cam,
                                                 max_repr_err=0.75, min_inliers=30, debug=(img1, img2, aflow, rel_q))

            if rel_q is not None:
                est_q = self.ori_est.estimate(yx1, yx2, matches, mask, depth1, img_rot1, (W1, H1), img_rot2, (W2, H2),
                                              cam, debug=(img1, img2, aflow, rel_q))

            if rel_q is not None and est_q is not None:
                ori_err = math.degrees(ds_tools.angle_between_q(est_q, rel_q))
                metrics = metrics + [ori_err, *est_q.components]
            elif rel_q is None:
                raise EvaluationException('Failed to estimate ground truth relative orientation',
                                          metrics + [np.nan] * 5)
            else:
                metrics = metrics + [np.inf] + [np.nan] * 4

        return metrics


class OrientationEstimator:
    def __init__(self, max_repr_err=1.5, min_inliers=25, use_ba=True, rotated_depthmaps=True,
                 debug=False, show_matches=False):
        self.max_repr_err = max_repr_err
        self.min_inliers = min_inliers
        self.use_ba = use_ba
        self.rotated_depthmaps = rotated_depthmaps
        self.debug = debug
        self.show_matches = show_matches
        self._debug_logger = fs_tools.get_logger("odo", level=logging.DEBUG) if self.debug > 1 else None

    def estimate_gt(self, aflow, depth1, img_rot1, size1, img_rot2, size2, cam: Camera, debug=None, **kwargs):
        tmp = self.__dict__.copy()
        for k, v in kwargs.items():
            setattr(self, k, v)

        unitflow = ds_tools.unit_aflow(aflow.shape[3], aflow.shape[2])
        xy1 = unitflow[torch.logical_not(torch.isnan(aflow[0, 0, :, :])).cpu().numpy()].reshape((-1, 2))

        # sparsify xy1
        if len(xy1) > 20000:
            n = len(xy1) // 10000
            xy1 = xy1[::n, :]

        xy2 = aflow[0, :, (xy1[:, 1] + 0.5).astype(int), (xy1[:, 0] + 0.5).astype(int)].cpu().numpy().T
        mask = np.logical_not(np.isnan(xy2[:, 0]))
        matches = np.arange(len(mask))

        gt_q = self.estimate(xy1, xy2, matches, mask, depth1, img_rot1, size1, img_rot2, size2, cam, debug=debug)

        for k in kwargs.keys():
            setattr(self, k, tmp[k])

        return gt_q

    def estimate(self, yx1, yx2, matches, mask, depth1, img_rot1, size1, img_rot2, size2, cam: Camera, debug=None):
        if isinstance(yx1, np.ndarray):
            xy1, xy2 = yx1, yx2
        else:
            xy1, xy2 = map(lambda yx: torch.flipud(yx[0, :, :]).t().cpu().numpy(), (yx1, yx2))
            matches, mask = map(lambda m: m[0, :].cpu().numpy(), (matches, mask))

        if np.sum(mask) < self.min_inliers:
            return None

        if debug is not None and (self.debug or self.show_matches):
            img1, img2, aflow, rel_q = debug

        if self.rotated_depthmaps:
            dist = nan_grid_interp(depth1, xy1[mask, :], max_radius=self.max_repr_err)

        # rotate keypoints back to original image coordinates
        xy1 = self.rotate_kps(xy1, -img_rot1, size1, (cam.width, cam.height))
        xy2 = self.rotate_kps(xy2, -img_rot2, size2, (cam.width, cam.height))

        if not self.rotated_depthmaps:
            dist = nan_grid_interp(depth1, xy1[mask, :], max_radius=self.max_repr_err)

        kp3d = cam.backproject(xy1[mask, :], dist=dist)
        repr_err_callback = (lambda kps, errs: self._plot_repr_errs(img2, kps, errs)) if self.show_matches else None
        pos, est_q, inliers = estimate_pose_pnp(cam, kp3d, xy2[matches[mask], :].astype(float),
                                                ransac=not self.use_ba, ba=self.use_ba,
                                                max_err=self.max_repr_err, input_in_opencv=True,
                                                repr_err_callback=repr_err_callback)
        if est_q is None or len(inliers) < self.min_inliers:
            return None

        if debug is not None and (self.debug or self.show_matches):
            ry, rp, rr = map(math.degrees, q_to_ypr(rel_q))
            ey, ep, er = map(math.degrees, q_to_ypr(est_q))
            logger.info('rel ypr: %.1f, %.1f, %.1f' % (ry, rp, rr))
            logger.info('est ypr: %.1f, %.1f, %.1f' % (ey, ep, er))

            if self.show_matches and 0:
                res_mask = np.zeros_like(mask)
                res_mask[mask][inliers] = True
                self._plot_matches(img1, xy1, img2, xy2, aflow, matches, res_mask)

        return est_q

    @staticmethod
    def rotate_kps(xy, angle, size0, size1):
        R = np.array([[math.cos(angle), -math.sin(angle)],
                      [math.sin(angle), math.cos(angle)]])
        rxy = (xy - np.array(size0).reshape((1, 2))/2).dot(R) + np.array(size1).reshape((1, 2))/2
        return rxy

    @staticmethod
    def _plot_repr_errs(img, xy, errs):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1)
        if 0:
            from .visualizations.misc import tensor2img
            img = cv2.cvtColor(tensor2img(img), cv2.COLOR_GRAY2RGB)
            axs[0].imshow(img)
        axs[0].scatter(xy[:, 0], xy[:, 1], np.clip(errs*30, 1, 10), facecolors='none', edgecolors='C0')
        axs[0].set_aspect('equal')
        axs[0].invert_yaxis()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_matches(img1, xy1, img2, xy2, aflow, matches, mask):
        img = OrientationEstimator._draw_matches(img1, xy1[mask, :], img2, xy2[matches[mask], :])

        gt_xy2 = aflow[0, :, (xy1[:, 1] + 0.5).astype(int), (xy1[:, 0] + 0.5).astype(int)].cpu().numpy().T[:, :]
        gt_mask = np.logical_and(mask, np.logical_not(np.isnan(gt_xy2[:, 0])))
        img_gt = OrientationEstimator._draw_matches(img1, xy1[gt_mask, :], img2, gt_xy2[gt_mask, :])

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
        axs[0].imshow(img)
        axs[1].imshow(img_gt)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _draw_matches(img1, xy1, img2, xy2):
        from .visualizations.misc import tensor2img
        img1, img2 = map(lambda img: cv2.cvtColor(tensor2img(img), cv2.COLOR_GRAY2RGB), (img1, img2))
        kp1, kp2 = map(lambda xy: [cv2.KeyPoint(x, y, 1) for x, y in xy.astype(float)], (xy1, xy2))
        matches = [cv2.DMatch(i, i, np.random.uniform(1, 2)) for i in range(len(kp1))]
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, singlePointColor=(0, 0, 255))
        return img


if __name__ == '__main__':
    main()

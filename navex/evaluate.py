import math
import argparse
import os.path

from tqdm import tqdm
import numpy as np
import quaternion
import cv2

import torch
from torch.utils.data import DataLoader

from .datasets.asteroidal.cg67p import CG67pOsinacPairDataset
from .datasets.asteroidal.eros import ErosPairDataset
from .datasets.asteroidal.itokawa import ItokawaPairDataset
from .datasets.asteroidal.synth import SynthBennuPairDataset
from .datasets.tools import Camera, q_to_ypr, from_opencv_q
from .datasets import tools as ds_tools
from .models import tools
from .extract import Extractor


# TODO:
#   - support training data metrics (map, mma, m-score, loc-err) for each sub-dataset (using cropping etc)


def main():
    parser = argparse.ArgumentParser("evaluate a feature extractor")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--output", type=str, required=True, help='csv file for evaluation output')
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
    args = parser.parse_args()

    ext = Extractor(args.model, gpu=args.gpu, top_k=args.top_k, feat_d=args.feat_d, border=args.border,
                    scale_f=args.scale_f, min_size=args.min_size, max_size=args.max_size,
                    min_scale=args.min_scale, max_scale=args.max_scale, det_lim=args.det_lim,
                    qlt_lim=args.qlt_lim, mode=args.det_mode, kernel_size=args.kernel_size)
    ori_est = OrientationEstimator()
    eval = ImagePairEvaluator(ext, ori_est, args.success_px_limit, args.mutual, args.ratio)

    datasets = {'eros': ErosPairDataset, '67p': CG67pOsinacPairDataset, 'ito': ItokawaPairDataset,
                'synth': SynthBennuPairDataset}
    datasets = {key: datasets[key] for key in args.dataset} if args.dataset else datasets

    write_header(args)

    for key, DatasetClass in datasets.items():
        dataset = DatasetClass(eval='test')
        pbar = tqdm(DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True), desc="Evaluating %s..." % key)
        for i, ((img1, img2), aflow, *meta) in enumerate(pbar):
            assert len(meta) >= 2 and dataset.cam, 'dataset "%s" does not have cam model or rel_q, rel_s metadata' % (key,)
            metrics = eval.evaluate(img1, img2, aflow, dataset.cam, np.quaternion(*meta[0].flatten()))
            write_row(args.output, key, dataset.samples[i][1], *dataset.samples[i][0], meta, metrics)

    # TODO:
    #   - eros, itokawa & osinac pairs have been selected with wrong criteria
    #   - fix databases for eros, itokawa, osinac
    #   - regenerate eros, itokawa, osinac pairs
    #   - regenerate synth dataset with no rotation except 180 deg lighting
    #   - compare r2d2 vs disk
    #   - decide if need to generate new synth dataset, maybe can just say there was some bug so that
    #     higher phase angles and 180 deg changes possible?
    #   - implement plots
    #   - run lafe optimization


def write_header(args):
    anames = [arg for arg in dir(args) if not arg.startswith('_') and arg not in ('output',)]
    avals = [getattr(args, name) for name in anames]
    header = ['Dataset', 'aflow', 'img1', 'img2',
              'light1_x', 'light1_y', 'light1_z', 'light2_x', 'light2_y', 'light2_z',
              'rel_qw', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_angle', 'rel_dist',
              'FD', 'M-Score', 'MMA', 'LE', 'mAP', 'ori-err', 'est_qw', 'est_qx', 'est_qy', 'est_qz']

    with open(args.output, 'w') as fh:
        fh.write('\t'.join(anames) + '\n')
        fh.write('\t'.join(map(str, avals)) + '\n\n')
        fh.write('\t'.join(header) + '\n')


def write_row(file, ds, aflow, img1, img2, meta, metrics):
    root = os.path.commonpath([aflow, img1, img2])
    rlen = len(root) + 1
    rel_q = np.quaternion(*meta[0].flatten().tolist())

    cam_axis = np.array([1, 0, 0])
    rel_angle = math.degrees(ds_tools.angle_between_v(cam_axis, ds_tools.q_times_v(rel_q, cam_axis)))

    with open(file, 'a') as fh:
        fh.write('\t'.join(map(str, (ds, aflow[rlen:], img1[rlen:], img2[rlen:],
                                     *meta[2].flatten().tolist(), *meta[3].flatten().tolist(),
                                     *rel_q.components, rel_angle, meta[1].item(), *metrics))) + '\n')


class ImagePairEvaluator:
    def __init__(self, extractor: Extractor, ori_est: 'OrientationEstimator', success_px_limit, mutual, ratio):
        self.extractor = extractor
        self.ori_est = ori_est
        self.success_px_limit = success_px_limit
        self.mutual = mutual
        self.ratio = ratio

    def evaluate(self, img1, img2, aflow, cam, rel_q):
        xys1, desc1, scores1 = self.extractor.extract(img1)
        xys2, desc2, scores2 = self.extractor.extract(img2)
        yx1 = torch.flipud(torch.tensor(xys1[:, :2].T, dtype=torch.long))[None, :, :]    # [K1, XYS] => [1, YX, K1]
        yx2 = torch.flipud(torch.tensor(xys2[:, :2].T, dtype=torch.long))[None, :, :]
        desc1 = torch.tensor(desc1.T)[None, :, :]     # [K1, D]   => [1, D, K1]
        desc2 = torch.tensor(desc2.T)[None, :, :]

        _, _, H1, W1 = img1.shape
        _, _, H2, W2 = img2.shape

        # [B, K1], [B, K1], [B, K1], [B, K1, K2]
        matches, norm, mask, dist = tools.match(desc1, desc2, mutual=self.mutual, ratio=self.ratio)

        brd2 = self.extractor.border * 2      # TODO: (4) exact mAP calculation  --v
        metrics = tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), self.success_px_limit,
                                      active_area=((H1 - brd2) * (W1 - brd2) + (H2 - brd2) * (W2 - brd2)) / 2)

        # calc relative orientation error (in opencv frame: +z cam axis, -y up)
        est_q = self.ori_est.estimate(yx1, yx2, matches, mask, cam)  #, debug=(img1, img2, aflow, rel_q))
        if est_q is not None:
            ori_err = math.degrees(ds_tools.angle_between_q(est_q, rel_q))
            metrics = metrics.flatten().tolist() + [ori_err, *est_q.components]
        else:
            metrics = metrics.flatten().tolist() + [np.nan] * 5

        return metrics


class OrientationEstimator:
    def __init__(self):
        self.ransac_p = 0.99999
        self.ransac_err = 1     # decrease?
        self.min_inliers = 30

    def estimate(self, yx1, yx2, matches, mask, cam: Camera, debug=None):
        xy1, xy2 = map(lambda yx: torch.flipud(yx[0, :, :]).t().cpu().numpy(), (yx1, yx2))
        matches, mask = map(lambda m: m[0, :].cpu().numpy(), (matches, mask))

        if debug is not None:
            img1, img2, aflow, rel_q = debug
            xy2 = aflow[0, :, (xy1[:, 1] + 0.5).astype(int), (xy1[:, 0] + 0.5).astype(int)].cpu().numpy().T
            mask = np.logical_not(np.isnan(xy2[:, 0]))
            matches = np.arange(len(mask))

        if np.sum(mask) < self.min_inliers:
            return None

        # undistort points
        xy1 = cam.undistort(xy1[mask, :].astype(float))
        xy2 = cam.undistort(xy2[matches[mask], :].astype(float))

        # solve pose using ransac & 5-point algo
        E, mask2 = cv2.findEssentialMat(xy1, xy2, cam.matrix, method=cv2.RANSAC,
                                        prob=self.ransac_p, threshold=self.ransac_err)

        if np.sum(mask2) < self.min_inliers:
            return None

        _, R, ur, mask3 = cv2.recoverPose(E, xy1, xy2, cam.matrix, mask=mask2.copy())
        inliers = np.where(mask3)[0]

        if len(inliers) < self.min_inliers or R is None:
            return None

        # in opencv: +z cam axis, -y up
        est_q = quaternion.from_rotation_matrix(R)

        # NOTE: currently est_q is very inaccurate even when giving a thousand ground truth correspondences
        # TODO: optimize with robust cost fun, use est_q as initial guess,
        #       include only inliers (?),
        #       discard outliers in a loop (?)

        # convert so that +x cam axis, +z up
        est_q = from_opencv_q(est_q)

        if debug is not None:
            img1, img2, aflow, rel_q = debug
            mask = mask3.flatten().astype(bool)
            gt_xy2 = aflow[0, :, (xy1[:, 0, 1] + 0.5).astype(int), (xy1[:, 0, 0] + 0.5).astype(int)] \
                          .cpu().numpy().T[:, None, :]

            ry, rp, rr = map(math.degrees, q_to_ypr(rel_q))
            ey, ep, er = map(math.degrees, q_to_ypr(est_q))
            print('\nrel ypr: %.1f, %.1f, %.1f' % (ry, rp, rr))
            print('est ypr: %.1f, %.1f, %.1f' % (ey, ep, er))

            img = self._draw_matches(img1, xy1, img2, xy2, mask)
            img_gt = self._draw_matches(img1, xy1, img2, gt_xy2,
                                        np.logical_and(mask, np.logical_not(np.isnan(gt_xy2[:, 0, 0]))))
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
            axs[0].imshow(img)
            axs[1].imshow(img_gt)
            plt.tight_layout()
            plt.show()

        return est_q

    @staticmethod
    def _draw_matches(img1, xy1, img2, xy2, mask):
        from .visualize import tensor2img
        img1, img2 = map(lambda img: cv2.cvtColor(tensor2img(img), cv2.COLOR_GRAY2RGB), (img1, img2))
        kp1, kp2 = map(lambda xy: [cv2.KeyPoint(x, y, 1) for x, y in xy[mask, 0, :]], (xy1, xy2))
        matches = [cv2.DMatch(i, i, np.random.uniform(1, 2)) for i in range(len(kp1))]
        img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, singlePointColor=(0, 0, 255))
        return img


if __name__ == '__main__':
    main()

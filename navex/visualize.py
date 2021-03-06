import os
import argparse

import cv2
import numpy as np
from torch import tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from navex.datasets.base import ExtractionImageDataset, RGB_STD, GRAY_STD, RGB_MEAN, GRAY_MEAN
from navex.extract import extract_multiscale
from navex.models import tools
from navex.models.tools import is_rgb_model, load_model


def main():
    parser = argparse.ArgumentParser("visualize features from images")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--images", type=str, required=True, help='image folder')
    parser.add_argument("--images2", type=str, help='second image folder')
    parser.add_argument("--images-ext", type=str, default='png', help='image extension')
    parser.add_argument("--tag", type=str, default='astr', help='feature file tag')
    parser.add_argument("--top-k", type=int, default=200, help='number of keypoints')
    parser.add_argument("--scale-f", type=float, default=2 ** (1/4))
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--det-lim", type=float, default=0.02)
    parser.add_argument("--qlt-lim", type=float, default=-10)
    parser.add_argument("--border", type=int, default=16, help="dont detect features if this close to image border")
    parser.add_argument("--min-matches", type=int, default=16)
    parser.add_argument("--skip-pose", type=int, default=0)
    parser.add_argument("--best-n", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    # --images data\batvik\2020-12-17\vid-1-frame-645.png
    # --images data\aachen\images_upright\db\125.jpg


    # TODO: make configurable
    dist_coefs = [-0.104090, 0.077530, -0.001243, -0.000088, 0.000000]
    cam_mx = np.array([[1580.356552, 0.000000, 994.026697],
                       [0.000000, 1580.553177, 518.938726],
                       [0.000000, 0.000000, 1.000000]]) * 0.5   # scaled to half the size
    cam_mx[2, 2] = 1

    device = "cuda:0" if args.gpu else "cpu"
    model = load_model(args.model, device)
    rgb = is_rgb_model(model)
    model.eval()

    dataset1 = ExtractionImageDataset(args.images, rgb=rgb)

    if args.images2:
        dataset2 = ExtractionImageDataset(args.images2, rgb=rgb)
        n1, n2 = len(dataset1), len(dataset2)
        inlier_counts = np.zeros((n1, n2))
        for i1 in range(n1):
            img1, xys1, desc1, scores1 = get_image_and_features(dataset1, i1, model, device, args)

            pbar = tqdm(range(n2))
            for i2 in pbar:
                if dataset1.samples[i1] != dataset2.samples[i2]:
                    img2, xys2, desc2, scores2 = get_image_and_features(dataset2, i2, model, device, args)
                    c = match(img1, xys1, desc1, img2, xys2, desc2, cam_mx, args, draw=False, pbar=pbar, device=device)
                    inlier_counts[i1, i2] = c
                else:
                    inlier_counts[i1, i2] = np.nan

            # plot inlier counts
            plt.figure(1)
            plt.plot(inlier_counts[i1, :])
            plt.title('inlier counts')
            plt.show()

            # best matches
            if 0:
                bst = np.argsort(-inlier_counts[i1, :])
            else:
                bst = [72, 73, 75]
            for k in range(args.best_n):
                print('%d: %s' % (inlier_counts[i1, bst[k]], dataset2.samples[bst[k]]))
                img2, xys2, desc2, scores2 = get_image_and_features(dataset2, bst[k], model, device, args)
                match(img1, xys1, desc1, img2, xys2, desc2, cam_mx, args, device=device)

    else:
        data_loader = model.wrap_ds(dataset1, shuffle=False)
        img0, xys0, desc0, scores0 = [None] * 4

        for i, data in enumerate(data_loader):
            img1 = ExtractionImageDataset.tensor2img(data)[0]
            data = data.to(device)

            xys1, desc1, scores1 = extract(model, data, args)

            if img0 is not None:
                match(img0, xys0, desc0, img1, xys1, desc1, cam_mx, args, device=device)

            img0, scores0, xys0, desc0 = img1, scores1, xys1, desc1


def get_image_and_features(dataset, idx, model, device, args):
    data1 = dataset[idx][None, :, :, :]
    img1 = ExtractionImageDataset.tensor2img(data1)[0]

    kpfile = dataset.samples[idx] + '.' + args.tag
    if os.path.exists(kpfile):
        xys1, desc1, scores1 = load_features(kpfile, device)
    else:
        xys1, desc1, scores1 = extract(model, data1.to(device), args)
        save_features(kpfile, data1.shape[2:], xys1, desc1, scores1)

    return img1, xys1, desc1, scores1


def save_features(kpfile, imsize, xys, desc, scores):
    with open(kpfile, 'wb') as fh:
        np.savez(fh, imsize=imsize,
                 keypoints=xys,
                 descriptors=desc,
                 scores=scores)


def load_features(kpfile, device):
    with open(kpfile, 'rb') as fh:
        tmp = np.load(fh)
        xys1, desc1, scores1 = [tmp[k] for k in ('keypoints', 'descriptors', 'scores')]
    if 0:
        desc1 = tensor(desc1).squeeze().permute((1, 0))[None, :, :].to(device)
    else:
        desc1 = tensor(desc1).to(device)
    return xys1, desc1, scores1


def extract(model, data, args):
    # extract keypoints/descriptors for a single image
    xys1, desc1, scores1 = extract_multiscale(model, data,
                                              scale_f=args.scale_f,
                                              min_scale=args.min_scale,
                                              max_scale=args.max_scale,
                                              min_size=args.min_size,
                                              max_size=args.max_size,
                                              top_k=args.top_k,
                                              det_lim=args.det_lim,
                                              qlt_lim=args.qlt_lim,
                                              border=args.border,
                                              verbose=True)

    scores1 = scores1.flatten()
    idxs = scores1.argsort()[-args.top_k or None:]
    scores1 = scores1[idxs]
    desc1 = np.swapaxes(desc1[idxs, :], 1, 0)[None, :, :]
    xys1 = xys1[idxs]
    return xys1, desc1, scores1


def match(img0, xys0, desc0, img1, xys1, desc1, cam_mx, args, draw=True, pbar=None, device=None):
    if isinstance(desc0, np.ndarray):
        desc0 = torch.Tensor(desc0).to(device)
    if isinstance(desc1, np.ndarray):
        desc1 = torch.Tensor(desc1).to(device)
    idxs, _, mask, _ = tools.match(desc0, desc1)
    idxs, mask = map(lambda x: x.cpu().numpy().flatten(), (idxs, mask))
    matches = np.array(list(zip(np.arange(len(mask))[mask], idxs[mask])))
    ini_matches = len(matches)

    ess_matches, pose_matches, inliers = 0, 0, []

    if ini_matches >= args.min_matches:
        # solve pose using ransac & 5-point algo
        E, mask = cv2.findEssentialMat(xys0[matches[:, 0], :2], xys1[matches[:, 1], :2], cam_mx,
                                       method=cv2.RANSAC, prob=0.99, threshold=8.0)
        ess_matches = np.sum(mask)
        if ess_matches >= args.min_matches and not args.skip_pose:
            _, R, ur, mask = cv2.recoverPose(E, xys0[matches[:, 0], :2], xys1[matches[:, 1], :2],
                                             cam_mx, mask=mask.copy())
            pose_matches = np.sum(mask)

        if pose_matches or args.skip_pose and ess_matches:
            inliers = np.where(mask)[0]

    if pbar:
        pbar.set_postfix({'k0':min(desc0.shape[2], desc1.shape[2]), 'k1':ini_matches,
                          'k2':ess_matches, 'k3':pose_matches})
    else:
        print('%d => %d => %d => %d' % (min(desc0.shape[2], desc1.shape[2]), ini_matches, ess_matches, pose_matches))

    # matches
    if draw and ini_matches:
        draw_matches(img0, xys0, img1, xys1, matches, scale=0.7, label='matches')

    # inliers
    if draw and len(inliers) > 0:
        draw_matches(img0, xys0, img1, xys1, [matches[i] for i in inliers], scale=0.7, label='inliers')

    return len(inliers)


def descriptor_change_map(des):
    dy = torch.linalg.norm(des[:, :, :-1, :] - des[:, :, 1:, :], dim=1)
    dx = torch.linalg.norm(des[:, :, :, :-1] - des[:, :, :, 1:], dim=1)
    dxy = dy[:, None, :, :-1] + dx[:, None, :-1, :]
    return dxy


def view_detections(imgs, dets, qlts, show=True):
    if not isinstance(imgs, (tuple, list)):
        imgs, dets, qlts = [imgs], [dets], [qlts]

    fig, axs = plt.subplots(len(imgs), 4, figsize=(12, 6), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, (img, det, qlt) in enumerate(zip(imgs, dets, qlts)):
        plot_tensor(img, image=True, ax=axs[i * 4 + 0])
        plot_tensor(img, image=True, heatmap=det * qlt, ax=axs[i * 4 + 1])
        plot_tensor(det, ax=axs[i * 4 + 2])
        plot_tensor(qlt, ax=axs[i * 4 + 3])

        if 0:
            xy = XY[-1] * sc
            for j in range(4):
                axs[i * 4 + j].plot(xy[:, 0], xy[:, 1], 'o', mfc='none')

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axs


def plot_tensor(data, heatmap=None, image=False, ax=None, scale=False):
    if data.dim() == 3:
        data = data[None, :, :, :]
    if heatmap is not None and heatmap.dim() == 3:
        heatmap = heatmap[None, :, :, :]
    B, D, H, W = data.shape
    for i in range(B):
        img = data[i, :, :, :].permute((1, 2, 0)).cpu().numpy()
        if image:
            if D == 3:
                img = img * np.array(RGB_STD, dtype=np.float32) + np.array(RGB_MEAN, dtype=np.float32)
            else:
                img = img * np.array(GRAY_STD, dtype=np.float32) + np.array(GRAY_MEAN, dtype=np.float32)
        elif scale:
            img = (img - img.min(axis=(0, 1))) / (img.max(axis=(0, 1)) - img.min(axis=(0, 1)))

        if image and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if heatmap is not None:
            overlay = heatmap[i, :, :, :].permute((1, 2, 0)).cpu().numpy()
            overlay = (overlay - overlay.min(axis=(0, 1))) / (overlay.max(axis=(0, 1)) - overlay.min(axis=(0, 1)))
            overlay = cv2.applyColorMap((overlay * 255).astype(np.uint8), cv2.COLORMAP_SUMMER)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            img = cv2.addWeighted(overlay, 0.5, (img * 255).astype(np.uint8), 0.5, 0)

        if ax is None:
            plt.imshow(img)
            plt.show()
        else:
            ax.imshow(img)


def draw_matches(img0, xys0, img1, xys1, matches, scale=1.0, pause=True, show=True, label='b) matches'):
    kp0 = [cv2.KeyPoint(x * scale, y * scale, s) for x, y, s in xys0]
    kp1 = [cv2.KeyPoint(x * scale, y * scale, s) for x, y, s in xys1]
    matches = list(cv2.DMatch(m[0], m[1], np.random.uniform(1, 2)) for m in matches)

    draw_params = {
#            matchColor: (88, 88, 88),
#        'matchColor': (-1, -1, -1),
        'singlePointColor': (0, 0, 255),
#            'flags': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    }

    # scale image, ensure rgb
    img0sc = cv2.cvtColor(cv2.resize(img0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
    img1sc = cv2.cvtColor(cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

    img3 = cv2.drawMatches(img0sc, kp0, img1sc, kp1, matches, None, **draw_params)

    if 0:
        cv2.imwrite('test.png', img3)

    if show:
        cv2.imshow(label, img3)
    cv2.waitKey(0 if pause else 25)


if __name__ == '__main__':
    main()

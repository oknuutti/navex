import os
import argparse
import pickle
import re
import warnings
from datetime import datetime

import cv2
import numpy as np
import quaternion
from torch.functional import F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from ..datasets.base import ExtractionImageDataset, RGB_STD, GRAY_STD, RGB_MEAN, GRAY_MEAN
from ..datasets.tools import find_files, angle_between_q
from ..extract import extract_multiscale, extract_traditional
from ..models import tools
from ..models.tools import is_rgb_model, load_model


def main():
    parser = argparse.ArgumentParser("visualize features from images")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--images", action='append', type=str, required=True, help='image folder')
    parser.add_argument("--images2", action='append', type=str, help='second image folder')
    parser.add_argument("--subdirs", type=int, default=0, help='recurse folders')
    parser.add_argument("--images-ext", type=str, default='png', help='image extension')
    parser.add_argument("--tag", type=str, default='astr', help='feature file tag')
    parser.add_argument("--top-k", type=int, default=None, help='number of keypoints')
    parser.add_argument("--feat-d", type=float, default=0.001, help='number of keypoints per pixel')
    parser.add_argument("--scale-f", type=float, default=2 ** (1/4))
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--det-lim", type=float, default=0.5)
    parser.add_argument("--qlt-lim", type=float, default=0.5)
    parser.add_argument("--border", type=int, default=16, help="dont detect features if this close to image border")
    parser.add_argument("--min-matches", type=int, default=16)
    parser.add_argument("--skip-pose", type=int, default=0)
    parser.add_argument("--best-n", type=int, default=5)
    parser.add_argument("--plot-indices", nargs='+', type=int, default=[], help="plot these image indices")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--video", type=int, default=0)
    parser.add_argument("--cluster-desc", action="store_true", help="cluster descriptors using mini-batch kmeans")
    parser.add_argument("--cluster-n", type=int, default=32, help="cluster count")
    parser.add_argument("--cluster-centers", help="save cluster centers to this file")
    parser.add_argument("--plot-clusters", action="store_true",
                        help="plot descriptor classification result based on cluster centers")
    parser.add_argument("--detection-only", action="store_true", help="only view feature detection overlay")
    args = parser.parse_args()

    # --images data\batvik\2020-12-17\vid-1-frame-645.png
    # --images data\aachen\images_upright\db\125.jpg

    if args.video:
        create_trajectory_video(args.images, args.images2, 'output/didy', args)

    # TODO: make configurable
    dist_coefs = [-0.104090, 0.077530, -0.001243, -0.000088, 0.000000]
    cam_mx = np.array([[1580.356552, 0.000000, 994.026697],
                       [0.000000, 1580.553177, 518.938726],
                       [0.000000, 0.000000, 1.000000]]) * 0.5   # scaled to half the size
    cam_mx[2, 2] = 1

    device = "cuda:0" if args.gpu else "cpu"
    traditional = args.model in ('akaze', 'orb', 'sift', 'surf')

    if not traditional:
        model = load_model(args.model, device)
        rgb = is_rgb_model(model)
        model.eval()
    else:
        model = args.model
        rgb = False

    dataset1 = ExtractionImageDataset(args.images, rgb=rgb, recurse=args.subdirs)

    if args.images2 and not args.detection_only and not args.cluster_desc and not args.plot_clusters:
        dataset2 = ExtractionImageDataset(args.images2, rgb=rgb, recurse=args.subdirs)
        n1, n2 = len(dataset1), len(dataset2)
        inlier_counts = np.zeros((n1, n2))
        match_counts = np.zeros((n1, n2))
        for i1 in range(n1):
            img1, xys1, desc1, scores1 = get_image_and_features(dataset1, i1, model, device, args)

            pbar = tqdm(range(n2))
            for i2 in pbar:
                if dataset1.samples[i1] != dataset2.samples[i2]:
                    img2, xys2, desc2, scores2 = get_image_and_features(dataset2, i2, model, device, args)
                    if len(xys2) > 0:
                        m, inl, rr = match(img1, xys1, desc1, img2, xys2, desc2, cam_mx, args, draw=False, pbar=pbar, device=device)
                        inlier_counts[i1, i2] = len(inl)
                        match_counts[i1, i2] = len(m)
                    else:
                        inlier_counts[i1, i2] = 0
                        match_counts[i1, i2] = 0
                else:
                    inlier_counts[i1, i2] = np.nan
                    match_counts[i1, i2] = np.nan

            # plot inlier counts
            fig = plt.figure(1)
            axs = fig.subplots(2, 1)
            axs[0].plot(match_counts[i1, :])   # / match_counts[i1, :])
            axs[0].set_title('match count')
            axs[1].plot(inlier_counts[i1, :])   # / match_counts[i1, :])
            axs[1].set_title('inlier ratio')
            plt.show()

            with open('output/temp/inlier-count-%s.pickle' % args.tag, 'wb') as fh:
                pickle.dump((inlier_counts, match_counts), fh)

            # best matches
            if not args.plot_indices:
                args.plot_indices = np.argsort(-inlier_counts[i1, :])
            else:
                # bst = [72, 73, 75]
                args.best_n = len(args.plot_indices)

            for k in range(args.best_n):
                print('%d: %s' % (args.plot_indices[k], dataset2.samples[args.plot_indices[k]]))
                img2, xys2, desc2, scores2 = get_image_and_features(dataset2, args.plot_indices[k], model, device, args)
                match(img1, xys1, desc1, img2, xys2, desc2, cam_mx, args, device=device,
                      save_img='output/temp/m_%s_%d.png' % (args.tag, args.plot_indices[k]))

    else:
        img0, xys0, desc0, scores0 = [None] * 4
        all_descs = []

        if args.plot_clusters and os.path.exists(args.cluster_centers):
            kmeans = load_kmeans(args.cluster_centers)
        else:
            do_matching = not (args.cluster_desc or args.plot_clusters) and not args.detection_only
            for i in tqdm(range(len(dataset1)), desc="Extracting features"):
                img1, xys1, desc1, scores1 = get_image_and_features(dataset1, i, model, device, args,
                                                                    as_tensor=do_matching)

                if args.cluster_desc or args.plot_clusters:
                    all_descs.append(np.squeeze(desc1).T)
                elif img0 is not None and not args.detection_only:
                    match(img0, xys0, desc0, img1, xys1, desc1, cam_mx, args, device=device)

                img0, scores0, xys0, desc0 = img1, scores1, xys1, desc1

            if args.cluster_desc or args.plot_clusters:
                all_descs = np.concatenate(all_descs, axis=0)
                kmeans = cluster(all_descs, args.cluster_n, args.cluster_centers)
                del all_descs

        if args.plot_clusters:
            assert kmeans is not None, 'failed to load/fit kmeans'
            data_loader = model.wrap_ds(dataset1, shuffle=True)
            for i, data in enumerate(data_loader):
                img = ExtractionImageDataset.tensor2img(data)[0]
                data = data.to(device)

                b, c, h0, w0 = data.shape
                sc = min(1, args.max_size / h0, args.max_size / w0)
                h, w = round(h0 * sc), round(w0 * sc)
                mode = 'bilinear' if sc > 0.5 else 'area'

                with torch.no_grad():
                    data = F.interpolate(data, (h, w), mode=mode, align_corners=False if mode == 'bilinear' else None)
                    des, det, qlt = model(data)
                des, det, qlt = map(lambda x: x.cpu().numpy(), (des, det, qlt))
                B, D, H, W = des.shape
                classes = kmeans.predict(np.moveaxis(des[0, :, :, :], 0, -1).reshape((-1, D))).reshape((H, W))
                overlay_classes(img, classes)


def overlay_classes(img, classes, ax=None):
    overlay = cv2.applyColorMap((classes * 255 / np.max(classes)).astype(np.uint8), cv2.COLORMAP_HSV)
    overlay = cv2.resize(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))
    if len(img.shape) < 3 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    if ax is None:
        plt.imshow(img)
        plt.show()
    else:
        ax.imshow(img)


def cluster(all_descs, n, save_file):
    from sklearn.cluster import MiniBatchKMeans

    np.random.shuffle(all_descs)
    if 0:
        cluster_num_plot(all_descs[:10000, :], range(2, 128))

    kmeans = MiniBatchKMeans(n_clusters=n,
                             init_size=4096 * 4,
                             batch_size=4096,
                             n_init=20,
                             max_iter=100,
                             random_state=42,
                             compute_labels=False,
                             verbose=1)
    kmeans.fit(all_descs)

    print('Saving cluster to %s...' % (save_file,))
    np.savez(save_file, (kmeans.cluster_centers_, np.array([n, 0, 0])))
    return kmeans


def cluster_num_plot(X, ns):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    ns = list(ns)
    sses = []
    sscores = []
    for n in tqdm(ns, desc='Clustering with varying n'):
        kmeans = KMeans(n_clusters=n, n_init=30, max_iter=300, random_state=42).fit(X)
        sscores.append(silhouette_score(X, kmeans.labels_))
        sses.append(kmeans.inertia_)

    plt.figure(1)
    plt.plot(ns, sses, label='SSE')
    plt.title('SSE')
    plt.figure(2)
    plt.plot(ns, sscores, label='Silhouette')
    plt.title('Silhouette')
    plt.show()


def load_kmeans(save_file):
    from sklearn.cluster import KMeans
    cluster_centers, stats = np.load(save_file, allow_pickle=True)['arr_0']
    n, inertia, sscore = stats
    print('Clustering (n=%d) SSE: %f, Silhouette Score: %f' % (n, inertia, sscore))
    kmeans = KMeans(cluster_centers)
    kmeans.cluster_centers_ = cluster_centers
    kmeans._n_threads = os.cpu_count()
    return kmeans


def get_image_and_features(dataset, idx, model, device, args, as_tensor=True):
    data1 = dataset[idx][None, :, :, :]
    img1 = ExtractionImageDataset.tensor2img(data1)[0]

    kpfile = dataset.samples[idx] + '.' + args.tag
    if os.path.exists(kpfile) and not args.detection_only:
        xys1, desc1, scores1 = load_features(kpfile)
        if args.top_k:
            xys1, scores1 = map(lambda x: x[:args.top_k, ...], (xys1, scores1))
            desc1 = desc1[:, :, :args.top_k]
    else:
        data1 = data1.to(device)
        for i in range(2):
            try:
                xys1, desc1, scores1 = extract(model, data1, args)
                break
            except Exception as e:
                if i == 0:
                    warnings.warn("Got %s, trying again using CPU" % (e,))
                    data1 = data1.cpu()
                    model = model.cpu()
                else:
                    raise e
        if i == 1:
            xys1, desc1, scores1 = map(lambda x: x.to(device), (xys1, desc1, scores1))

        save_features(kpfile, data1.shape[2:], xys1, desc1, scores1)

    if as_tensor and not isinstance(desc1, torch.Tensor):
        # desc1 = torch.Tensor(desc1).squeeze().permute((1, 0))[None, :, :].to(device)
        desc1 = torch.Tensor(desc1).to(device)

    return img1, xys1, desc1, scores1


def save_features(kpfile, imsize, xys, desc, scores):
    with open(kpfile, 'wb') as fh:
        np.savez(fh, imsize=imsize,
                 keypoints=xys,
                 descriptors=desc,
                 scores=scores)


def load_features(kpfile):
    with open(kpfile, 'rb') as fh:
        tmp = np.load(fh)
        xys1, desc1, scores1 = [tmp[k] for k in ('keypoints', 'descriptors', 'scores')]
    return xys1, desc1, scores1


def extract(model, data, args):
    # extract keypoints/descriptors for a single image
    if isinstance(model, str):
        assert tuple(data.shape[0:2]) == (1, 1), 'image tensor shape %s not supported' % (data.shape,)
        assert data.device.type == 'cpu', 'need to run on cpu'
        img = data.numpy()[0, 0, :, :]
        img = img * np.array(GRAY_STD, dtype=np.float32) + np.array(GRAY_MEAN, dtype=np.float32)
        img = np.clip(img*255 + 0.5, 0, 255).astype(np.uint8)

        xys1, desc1, scores1 = extract_traditional(model, img,
                                                   asteroid_target=True,
                                                   top_k=args.top_k,
                                                   feat_d=args.feat_d,
                                                   border=args.border)
    else:
        xys1, desc1, scores1 = extract_multiscale(model, data,
                                                  scale_f=args.scale_f,
                                                  min_scale=args.min_scale,
                                                  max_scale=args.max_scale,
                                                  min_size=args.min_size,
                                                  max_size=args.max_size,
                                                  top_k=args.top_k,
                                                  feat_d=args.feat_d,
                                                  det_lim=args.det_lim,
                                                  qlt_lim=args.qlt_lim,
                                                  border=args.border,
                                                  verbose=True,
                                                  plot=args.detection_only)

    scores1 = scores1.flatten()
    idxs = (-scores1).argsort()[:args.top_k]
    scores1 = scores1[idxs]
    desc1 = np.swapaxes(desc1[idxs, :], 1, 0)[None, :, :]
    xys1 = xys1[idxs]
    return xys1, desc1, scores1


def match(img0, xys0, desc0, img1, xys1, desc1, cam_mx, args, draw=True, pbar=None, device=None, save_img=None):
    if isinstance(desc0, np.ndarray):
        desc0 = torch.Tensor(desc0).to(device)
    if isinstance(desc1, np.ndarray):
        desc1 = torch.Tensor(desc1).to(device)
    idxs, _, mask, _ = tools.match(desc0, desc1, norm='hamming' if desc0.dtype == torch.uint8 else 2)
    idxs, mask = map(lambda x: x.cpu().numpy().flatten(), (idxs, mask))
    matches = np.array(list(zip(np.arange(len(mask))[mask], idxs[mask])))
    ini_matches = len(matches)
    rel_rot = None

    ess_matches, pose_matches, inliers = 0, 0, []

    if ini_matches >= args.min_matches:
        # solve pose using ransac & 5-point algo
        E, mask = cv2.findEssentialMat(xys0[matches[:, 0], :2], xys1[matches[:, 1], :2], cam_mx,
                                       method=cv2.RANSAC, prob=0.999, threshold=2.0)
        ess_matches = np.sum(mask)
        if ess_matches >= args.min_matches and not args.skip_pose:
            _, rel_rot, ur, mask = cv2.recoverPose(E, xys0[matches[:, 0], :2], xys1[matches[:, 1], :2],
                                             cam_mx, mask=mask.copy())
            pose_matches = np.sum(mask)

        if pose_matches or args.skip_pose and ess_matches:
            inliers = np.where(mask)[0]

    if pbar:
        pbar.set_postfix({'k0': min(desc0.shape[2], desc1.shape[2]), 'k1': ini_matches,
                          'k2': ess_matches, 'k3': pose_matches})
    else:
        print('%d => %d => %d => %d' % (min(desc0.shape[2], desc1.shape[2]), ini_matches, ess_matches, pose_matches))

    # matches
    if draw and ini_matches:
        mtc_img = draw_matches(img0, xys0, img1, xys1, matches, height=512, return_only=len(inliers) > 0, label='matches')

    # inliers
    if draw and len(inliers) > 0:
        inl_img = draw_matches(img0, xys0, img1, xys1, [matches[i] for i in inliers],
                               height=512, return_only=True, show_orig=False, label='inliers')
        img = np.concatenate((mtc_img, inl_img), axis=0)
        if save_img is None:
            cv2.imshow('inliers', img)
            cv2.waitKey()
        else:
            cv2.imwrite(save_img, img)

    return matches, [matches[i] for i in inliers], rel_rot


def descriptor_change_map(des):
    dy = torch.linalg.norm(des[:, :, :-1, :] - des[:, :, 1:, :], dim=1)
    dx = torch.linalg.norm(des[:, :, :, :-1] - des[:, :, :, 1:], dim=1)
    dxy = dy[:, None, :, :-1] + dx[:, None, :-1, :]
    return dxy


def view_detections(imgs, dets, qlts, show=True, title=None):
    if not isinstance(imgs, (tuple, list)):
        imgs, dets, qlts = [imgs], [dets], [qlts]

    fig, axs = plt.subplots(len(imgs), 4, figsize=(12, 6), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, (img, det, qlt) in enumerate(zip(imgs, dets, qlts)):
        if qlt.shape[-2:] != img.shape[-2:]:
            qlt = F.interpolate(qlt[None, ...], size=img.shape[-2:], mode='nearest')[0, ...]
        plot_tensor(img, image=True, ax=axs[i * 4 + 0])
        plot_tensor(img, image=True, heatmap=det * qlt, ax=axs[i * 4 + 1])
        plot_tensor(heatmap=det, ax=axs[i * 4 + 2])
        plot_tensor(heatmap=qlt, ax=axs[i * 4 + 3])

        if 0:
            xy = XY[-1] * sc
            for j in range(4):
                axs[i * 4 + j].plot(xy[:, 0], xy[:, 1], 'o', mfc='none')

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axs


def tensor2img(img, i=0):
    if len(img.shape) == 4:
        img = img[i, :, :, :]
    D, H, W = img.shape
    img = img.permute((1, 2, 0)).cpu().numpy()

    if D == 3:
        img = img * np.array(RGB_STD, dtype=np.float32) + np.array(RGB_MEAN, dtype=np.float32)
    else:
        img = img * np.array(GRAY_STD, dtype=np.float32) + np.array(GRAY_MEAN, dtype=np.float32)

    return (img * 255).astype(np.uint8)


def plot_tensor(data=None, heatmap=None, image=False, ax=None, scale=False, color_map='hsv'):
    if data is not None and data.dim() == 3:
        data = data[None, :, :, :]
    if heatmap is not None and heatmap.dim() == 3:
        heatmap = heatmap[None, :, :, :]
    assert data is not None or heatmap is not None, 'both data and heatmap are None, give at least one'
    B, D, H, W = heatmap.shape if data is None else data.shape

    for i in range(B):
        if data is not None:
            img = data[i, :, :, :].permute((1, 2, 0)).cpu().numpy()
            if image:
                if D == 3:
                    img = img * np.array(RGB_STD, dtype=np.float32) + np.array(RGB_MEAN, dtype=np.float32)
                else:
                    img = img * np.array(GRAY_STD, dtype=np.float32) + np.array(GRAY_MEAN, dtype=np.float32)
            elif scale:
                val_range = img.max(axis=(0, 1)) - img.min(axis=(0, 1))
                img = (img - img.min(axis=(0, 1))) / (1 if val_range == 0 else val_range)

            if image and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)

        if heatmap is not None:
            overlay = heatmap[i, :, :, :].permute((1, 2, 0)).cpu().numpy()
            vmin, vmax = np.quantile(overlay, [0.01, 0.99], axis=(0, 1))
            overlay = (overlay - vmin) / (1 if vmax - vmin == 0 else vmax - vmin)

            if color_map == 'hsv':
                s, c = 0.33, 0.10
                overlay = cv2.applyColorMap(((1 - np.clip(overlay*(1-s+c)+s-c, s, 1.0))*255).astype(np.uint8), cv2.COLORMAP_HSV)
            else:
                overlay = cv2.applyColorMap((np.clip(overlay, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_SUMMER)

            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            if data is None:
                img = overlay
            else:
                img = cv2.addWeighted(overlay, 0.3, (img * 255).astype(np.uint8), 0.7, 0)

        if ax is None:
            plt.imshow(img)
            plt.show()
        else:
            if not isinstance(ax, (tuple, list, np.ndarray)):
                ax = [ax]
            ax[i].imshow(img)


def draw_matches(img0, xys0, img1, xys1, matches, height=None, pause=True, show=True,
                 show_orig=True, return_only=False, label='b) matches'):
    if height is None:
        scale = 1.0
        rel_sc = img1.shape[0] / img0.shape[0]
        sc0 = scale * (rel_sc if rel_sc <= 1 else 1.0)
        sc1 = scale * (1/rel_sc if rel_sc > 1 else 1.0)
    else:
        sc0 = height / img0.shape[0]
        sc1 = height / img1.shape[0]

    kp0 = [cv2.KeyPoint(x * sc0, y * sc0, s) for x, y, s in xys0]
    kp1 = [cv2.KeyPoint(x * sc1, y * sc1, s) for x, y, s in xys1]
    matches = list(cv2.DMatch(m[0], m[1], np.random.uniform(1, 2)) for m in matches)

    draw_params = {
#            matchColor: (88, 88, 88),
#        'matchColor': (-1, -1, -1),
        'singlePointColor': (0, 0, 255),
#            'flags': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    }

    # scale image, ensure rgb
    img0sc = cv2.cvtColor(cv2.resize(img0, None, fx=sc0, fy=sc0, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
    img1sc = cv2.cvtColor(cv2.resize(img1, None, fx=sc1, fy=sc1, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)

    img3 = cv2.drawMatches(img0sc, kp0, img1sc, kp1, matches, None, **draw_params)

    if show_orig:
        tmp = np.concatenate((img0sc, img1sc), axis=1)
        img3 = np.concatenate((tmp, img3), axis=0)

    if 0 and not return_only:
        cv2.imwrite('test.png', img3)

    if show and not return_only:
        cv2.imshow(label, img3)
        cv2.waitKey(0 if pause else 25)

    return img3


def plot_inlier_counts(folder, prefix='inlier-count-', postfix='.pickle'):
    data = {}
    s, e = len(prefix), len(postfix)
    for file in os.listdir(folder):
        if file[-e:] == postfix:
            with open(os.path.join(folder, file), 'rb') as fh:
                data[file[s:-e]] = pickle.load(fh)

    if 0:
        fig, axs = plt.subplots(2, 1)
        for tag, d in data.items():
            axs[0].plot(d[1].flatten() / np.nanmax(d[1]), label=tag)
            axs[1].plot((d[0] / d[1]).flatten(), label=tag)
        axs[0].set_title('matches')
        axs[1].set_title('inlier ratio')
        axs[0].legend()
        plt.tight_layout()
    elif 1:
        mapping = {"akaze": (0, "AKAZE"), "hafe_ldisk_184": (1, "HAFE")}
        fig, axs = plt.subplots(len(mapping), 1)
        data = [[o, l, data[k]] for k, (o, l) in mapping.items()]
        data = sorted(data, key=lambda x: x[0])

        for i, label, d in data:
            axs[i].set_title('%s' % label)
            axs[i].plot(d[1].flatten(), label="Matches")
            axs[i].plot(d[0].flatten(), label="Inliers")
            axs[i].legend()

        plt.tight_layout()
    else:
        for tag, d in data.items():
            plt.figure()
            plt.plot(d[0].flatten())
            plt.title('inlier counts')
            plt.tight_layout()
    plt.show()


def create_trajectory_video(state_file, image_folder, out_folder, args):
    # read simulation output for later drawing trajectory image
    with open(state_file) as fh:
        raw = []
        for line in fh:
            raw.append(list(map(float, line.split(','))))
    state_data = np.array(raw)
    st_t = state_data[:, 0]
    st_sc_loc = state_data[:, 1:4]
    st_sc_q = quaternion.from_float_array(state_data[:, 4:8])
    st_d1_loc = state_data[:, 8:11]
    st_d1_q = quaternion.from_float_array(state_data[:, 11:15])
    st_d2_loc = state_data[:, 15:18]
    st_d2_q = quaternion.from_float_array(state_data[:, 18:22])

    # cam matrix for geometric feature match validation
    cam_mx = np.array([[1.5216246e+04, 0.0000000e+00, 1.0240000e+03],
                       [0.0000000e+00, 1.5218485e+04, 9.7200000e+02],
                       [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])

    # requirements for image match
    min_inliers, min_inl_ratio, min_delay = 30, 0.5, 2.0 * 60 * 60

    # load feature extraction cnn model
    device = "cuda:0"
    model = load_model(args.model, device)
    rgb = is_rgb_model(model)
    model.eval()

    # load dataset
    dataset = ExtractionImageDataset(image_folder, rgb=rgb, recurse=False, regex=r'\d.png$')
    n = len(dataset)
    pbar = tqdm(range(n))
    previous = []
    idxs = []
    img_inl = None
    match_idx = None
    match_err = None

    for i1 in pbar:
        # extract features
        img_main1, xys1, desc1, scores1 = get_image_and_features(dataset, i1, model, device, args)

        img_file = dataset.samples[i1].split(os.path.sep)[-1]
        isotime = re.sub(r'(\d{4}-\d{2}-\d{2}T)(\d{2})(\d{2})(\d{2})', r'\1\2:\3:\4', img_file[:-4])
        img_t1 = datetime.fromisoformat(isotime).timestamp() + 2*3600  # some timezone thing?
        idx1 = np.where(st_t >= img_t1)[0][0]
        previous.append((img_t1, img_main1, xys1, desc1, scores1))
        idxs.append(idx1)

        for i2 in range(i1):
            # search for previous matching scenes
            img_t2, img_main2, xys2, desc2, scores2 = previous[i2]
            if img_t1 - img_t2 > min_delay:
                matches, inliers, rel_rot = match(img_main1, xys1, desc1, img_main2, xys2, desc2, cam_mx,
                                                  args, draw=False, pbar=pbar, device=device)

                if len(inliers) > min_inliers and len(inliers) / len(matches) > min_inl_ratio:
                    # draw feature matches
                    img_inl = draw_matches(img_main1, xys1, img_main2, xys2, inliers, height=img_main1.shape[0]//3,
                                           return_only=True, show_orig=False, label='inliers')
                    match_idx = idxs[i2]
                    gt_sc_d1_q1 = st_sc_q[idx1].conj() * st_d1_q[idx1]
                    gt_sc_d1_q2 = st_sc_q[match_idx].conj() * st_d1_q[match_idx]
                    gt_dq = gt_sc_d1_q1.conj() * gt_sc_d1_q2
                    match_err = angle_between_q(quaternion.from_rotation_matrix(rel_rot), gt_dq)
                    pbar.set_postfix({'match_err': round(match_err / np.pi * 180, 1)})
                    # TODO: debug match_err, if can get to work, incl in result image
                    #   - probably need to convert from simulation coordinate frame convention to the opencv convention
                    break

        if img_inl is None:
            img_inl = np.zeros((img_main1.shape[0]//3, 2*img_main1.shape[1]//3, 3), dtype=np.uint8)

        # draw trajectory
        bg_col, fg_col = (0, 0, 0), (150/255, 200/255, 251/255)
        h, w = img_main1.shape[0] - img_inl.shape[0], img_inl.shape[1]
        fig = plt.figure(figsize=(8, round(8*h/w, 1)))
        plt.plot(st_sc_loc[idxs, 0], st_sc_loc[idxs, 1], 'C0')
        plt.plot(st_d1_loc[idxs, 0], st_d1_loc[idxs, 1], 'C1')
        plt.plot(st_d2_loc[idxs, 0], st_d2_loc[idxs, 1], 'C2')
        plt.plot(st_sc_loc[idxs[-1], 0], st_sc_loc[idxs[-1], 1], 'oC0', mfc='none')
        plt.plot(st_d1_loc[idxs[-1], 0], st_d1_loc[idxs[-1], 1], 'oC1', mfc='none')
        plt.plot(st_d2_loc[idxs[-1], 0], st_d2_loc[idxs[-1], 1], 'oC2', mfc='none')

        if match_idx is not None:
            plt.plot(st_sc_loc[match_idx, 0], st_sc_loc[match_idx, 1], 'xC0', mfc='none')

        fig.patch.set_facecolor(bg_col)
        fig.gca().set_facecolor(bg_col)
        fig.gca().spines['bottom'].set_color(fg_col)
        fig.gca().spines['left'].set_color(fg_col)
        fig.gca().xaxis.label.set_color(fg_col)
        fig.gca().yaxis.label.set_color(fg_col)
        fig.gca().tick_params(colors=fg_col)

        plt.xlim((np.min(st_sc_loc[:, 0]), np.max(st_sc_loc[:, 0])))
        plt.ylim((np.min(st_sc_loc[:, 1]), np.max(st_sc_loc[:, 1])))
        plt.gca().axis('equal')
        plt.tight_layout()

        fig.canvas.draw()
        img_tmp = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w_, h_ = np.array(fig.canvas.devicePixelRatio()) * fig.canvas.get_width_height()
        plt.close(fig)

        h_ = int(img_tmp.size/3/w_)  # for some reason h_ isn't always correct
        img_tmp = np.flip(img_tmp.reshape((h_, w_, 3)), axis=2)
        img_tmp = cv2.resize(img_tmp, None, fx=h/h_, fy=h/h_)
        img_traj = np.zeros((h, w, 3), dtype=np.uint8)
        img_traj[:min(img_tmp.shape[0], h), :min(img_tmp.shape[1], w), :] = img_tmp[:h, :w, :]

        # compose final image and save it
        img_right = np.concatenate((img_inl, img_traj), axis=0)
        img = np.concatenate((cv2.cvtColor(img_main1, cv2.COLOR_GRAY2BGR), img_right), axis=1)
        cv2.imwrite(os.path.join(out_folder, img_file), img)


if __name__ == '__main__':
    if 1:
        main()
    else:
        plot_inlier_counts('output/temp')

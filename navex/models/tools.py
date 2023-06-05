import math
import os
from collections import OrderedDict

import numpy as np
import scipy
import scipy.stats
import scipy.optimize

import torch
from torch import nn
from torch.functional import F


def calc_padding(tensor, div):
    hpad = (-tensor.shape[-1]) % div
    vpad = (-tensor.shape[-2]) % div
    l = hpad // 2
    r = l + hpad % 2
    t = vpad // 2
    b = t + vpad % 2
    padding = [l, r, t, b]
    return padding


def detect_from_dense(des, det, qlt, top_k=None, feat_d=0.001, det_lim=0.02, qlt_lim=0.02,
                      border=16, kernel_size=3, mode='nms', interp='bilinear', use_grid_sample=True):
    B, D, Hs, Ws = des.shape
    _, _, Ht, Wt = det.shape
    _, _, Hq, Wq = qlt.shape

    des_shape_mismatch = (Hs, Ws) != (Ht, Wt)

    # interpolate if different scale heads
    if des_shape_mismatch and not use_grid_sample:
        des = F.interpolate(des, (Ht, Wt), mode=interp, align_corners=True)
    if (Hq, Wq) != (Ht, Wt):
        qlt = F.interpolate(qlt, (Ht, Wt), mode=interp, align_corners=True)

    # filter to remove high freq, likely spurious detections
    det = F.avg_pool2d(det, kernel_size=3, stride=1, padding=1)

    if mode == 'nms':
        # local maxima
        max_filter = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=1)
        maxima = (det == max_filter(det))  # [b, 1, h, w]
    elif mode == 'grid':
        # gridded maxima
        hpad = -((det.shape[-1] - 2*border + kernel_size//2) % kernel_size - kernel_size//2)
        vpad = -((det.shape[-2] - 2*border + kernel_size//2) % kernel_size - kernel_size//2)
        assert hpad < border and vpad < border, \
            f'invalid gridded detection kernel_size={kernel_size}, w={det.shape[-1]}, h={det.shape[-2]}, b={border}'
        cdet = F.pixel_unshuffle(det[:, :, border:-border+vpad, border:-border+hpad], kernel_size)
        cmaxima = (cdet == torch.max(cdet, dim=1, keepdim=True)[0])
        maxima = torch.zeros_like(det, dtype=torch.bool)
        maxima[:, :, border:-border+vpad, border:-border+hpad] = F.pixel_shuffle(cmaxima, kernel_size)
        del cdet, cmaxima

        # remove double detections at grid borders
        sdet = det * maxima
        max_filter = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        maxima = (det == max_filter(sdet))  # [b, 1, h, w]
        del sdet
    else:
        assert False, f'invalid mode="{mode}"'

    # remove low confidence detections
    maxima *= (det >= det_lim)
    maxima *= (qlt >= qlt_lim)

    # remove detections at the border
    maxima[:, 0, :border, :] = False
    maxima[:, 0, -border:, :] = False
    maxima[:, 0, :, :border] = False
    maxima[:, 0, :, -border:] = False

    K = maxima.sum(dim=(2, 3)).max().item()

    if feat_d is not None:
        # detect at most 0.001 features per pixel
        K = min(K, int((Ht - border * 2) * (Wt - border * 2) * feat_d))

    if top_k is not None:
        K = min(K, top_k)

    yx = -torch.ones((B, 2, K), device=des.device, dtype=torch.long)
    scores = torch.zeros((B, 1, K), device=des.device, dtype=qlt.dtype)
    descr = torch.zeros((B, D, K), device=des.device, dtype=des.dtype)
    for b in range(B):
        # get indices for local maxima
        idxs = maxima[b, 0, :, :].nonzero(as_tuple=True)      # [k, 2]
        k = min(len(idxs[0]), K)

        # calculate scores, sort by them
        sc = det[b, 0, idxs[0], idxs[1]] * qlt[b, 0, idxs[0], idxs[1]]
        sc, ord = torch.sort(sc, descending=True)
        idxs = (idxs[0][ord[:k]], idxs[1][ord[:k]])

        # get scores and descriptors at local maxima
        yx[b, 0, 0:k] = idxs[0]
        yx[b, 1, 0:k] = idxs[1]
        scores[b, 0, 0:k] = sc[:k]

        if des_shape_mismatch and use_grid_sample:
            I = torch.stack([2 * idxs[1][None, None, :].float() / Wt - 1,
                             2 * idxs[0][None, None, :].float() / Ht - 1], dim=-1)
            descr[b:b+1, :, 0:k] = F.grid_sample(des[b:b+1, :, :, :], I, align_corners=True, mode=interp).squeeze(2)
        else:
            descr[b:b+1, :, 0:k] = des[b:b+1, :, idxs[0], idxs[1]]

    return yx, scores, descr


class MatchException(Exception):
    pass


def scale_restricted_match(syx1, des1, syx2, des2, norm=2, mutual=True, ratio=0, octave_levels=4, type='hires'):
    # log scales used
    s1, s2 = syx1[:, 0, :].log(), syx2[:, 0, :].log()

    # initial matching for scale difference estimation
    matches, mdist, mask, dist = match(des1, des2, norm, mutual, ratio)
    B, K1, K2 = dist.shape
    m1, m2 = [], []

    # one level scale difference
    lvl_sc = np.log(2) / octave_levels
    match_levels = 1
    match_margin = 0.6

    def group_sd(sd):
        x, y = np.unique(sd, return_counts=True)
        arr = []    # [sum(x*y), sum(y)]
        for i in range(len(x)):
            if i > 0 and abs(x[i] - arr[-1][0] / arr[-1][1]) < lvl_sc * 0.1:
                arr[-1][0] += x[i]*y[i]
                arr[-1][1] += y[i]
            else:
                arr.append([x[i]*y[i], y[i]])
        return np.array([[sx/sy, sy] for sx, sy in arr]).reshape((-1, 2)).T

    for b in range(B):
        # get scale difference
        ms1, ms2 = s1[b, mask[b, :]], s2[b, matches[b, mask[b, :]]]
        sd = (ms2 - ms1).cpu().numpy()

        # gaussian kernel density estimate
        try:
            kde = scipy.stats.gaussian_kde(sd, bw_method=3 * lvl_sc)
        except np.linalg.LinAlgError as e:
            raise MatchException('Gaussian KDE failed') from e
        sd_mean = np.mean(sd)

        # get mode, start from the mean
        sd_mode = scipy.optimize.minimize_scalar(lambda x: -kde(x), method='bounded',
                                                 bounds=(sd_mean - 1.5 * lvl_sc, sd_mean + 1.5 * lvl_sc)).x

        if 0:
            s, n = group_sd(sd)
            x = np.linspace(sd.min(), sd.max(), 1000)

            import matplotlib.pyplot as plt
            plt.plot(x, kde(x))
            plt.plot(s, n / n.max() * kde(sd_mode), 'x')
            plt.show()

        if type == 'hires':
            cs2 = s2[b, :] - sd_mode
            s1_min, s2_min = s1[b, :].min(), cs2.min()

            # take the highest resolution features from the lowest resolution image, match those to similar scale features
            if s2_min > s1_min:
                # second image has lower resolution, select only top level features
                mask2 = cs2 <= s2_min + lvl_sc * (match_levels - 1 + match_margin)

                # match only features with similar scale
                mask1 = (s1[b, :] - cs2[mask2].mean()).abs() < lvl_sc * (match_levels - 1 + match_margin)

            else:
                # first image has lower resolution, select only top level features
                mask1 = s1[b, :] <= s1_min + lvl_sc * (match_levels - 1 + match_margin)

                # match only features with similar scale
                mask2 = (cs2 - s1[b, mask1].mean()).abs() < lvl_sc * (match_levels - 1 + match_margin)

            match_mask = None

        elif type == 'weighted':
            match_mask = (s1.view((1, K1, 1)).expand((1, K1, K2)) + sd_mode
                          - s2.view((1, 1, K2)).expand((1, K1, K2))).abs() / (lvl_sc * (match_levels - 1 + 0.6)) + 1
            match_mask = match_mask.type(torch.float32)
            mask1 = torch.ones((K1,), device=des1.device, dtype=torch.bool)
            mask2 = torch.ones((K2,), device=des1.device, dtype=torch.bool)

        else:
            assert type == 'windowed', 'Unknown match type: %s' % type
            match_mask = (s1.view((1, K1, 1)).expand((1, K1, K2)) + sd_mode
                          - s2.view((1, 1, K2)).expand((1, K1, K2))).abs() < lvl_sc * (match_levels - 1 + 0.6)
            mask1 = match_mask.any(dim=2).view(-1)
            mask2 = match_mask.any(dim=1).view(-1)
            match_mask = match_mask[:, mask1, :][:, :, mask2]

        # [B, K1], [B, K1], [B, K1], [B, K1, K2]
        _matches, _mdist, _mask, _dist = match(des1[b:b+1, :, mask1], des2[b:b+1, :, mask2], norm, mutual, ratio,
                                               mask=match_mask)

        if mask1.sum() > 0 and mask2.sum() > 0:
            matches[b:b + 1, mask1] = _matches
            mdist[b:b + 1, mask1] = _mdist
            mask[b:b + 1, :] = False
            mask[b:b + 1, mask1] = _mask
            dist[(slice(b, b + 1),) + np.ix_(mask1, mask2)] = _dist
        else:
            s, n = group_sd(sd)
            raise MatchException(f'No features pass scale restriction, details:'
                                 f' n1: {mask1.sum()}, n2: {mask2.sum()}, sd_mode: {sd_mode}, sd_mean: {sd_mean},'
                                 f' s: {s.tolist()}, n: {n.tolist()}')
        m1.append(mask1)
        m2.append(mask2)

    return matches, mdist, mask, dist, torch.stack(m1), torch.stack(m2)


def match(des1, des2, norm=2, mutual=True, ratio=0, mask=None):
    B, D, K1 = des1.shape
    _, _, K2 = des2.shape
    dev = des1.device

    if K1 == 0 or K2 == 0:
        return torch.zeros((B, 2, K1), dtype=torch.long, device=dev), \
               torch.zeros((B, K1), dtype=torch.float, device=dev), \
               torch.zeros((B, K1), dtype=torch.bool, device=dev), \
               torch.zeros((B, K1, K2), dtype=torch.float, device=dev)

    if isinstance(norm, int):
        dist = torch.linalg.norm(des1.view((B, D, K1, 1)).expand((B, D, K1, K2))
                                 - des2.view((B, D, 1, K2)).expand((B, D, K1, K2)), ord=norm, dim=1)     # [b, k1, k2]
    else:
        assert des1.shape[0] == 1 and des2.shape[0] == 1, 'with binary descriptors, currently only one image at a time supported'
        assert des1.dtype == torch.uint8 and des2.dtype == torch.uint8, 'binary descriptors need to be of type byte'
        assert des1.device.type == 'cpu', 'only works for cpu at the moment'
        des1_ = des1.numpy()[0, :, :].T
        des2_ = des2.numpy()[0, :, :].T
        d1 = np.repeat(des1_[:, None, :], len(des2_), axis=1)
        d2 = np.repeat(des2_[None, :, :], len(des1_), axis=0)

        d1 = np.unpackbits(d1, axis=2)
        d2 = np.unpackbits(d2, axis=2)
        dist = np.sum(d1 != d2, axis=2)
        dist = torch.Tensor(dist[None, :, :])

    if mask is not None:
        if mask.dtype in (torch.bool, torch.uint8):
            dist[~mask] = float('inf')
        else:
            dist = dist * mask

    min1, idx1 = torch.min(dist, dim=2)
    mask = torch.ones((B, K1), dtype=torch.bool, device=dev)

    if mutual:
        # check that matches are mutually closest
        min2, idx2 = torch.min(dist, dim=1)
        for b in range(B):
            mask[b, :] *= (idx2[b, idx1[b, :]] == torch.arange(0, K1, device=dev))

    if ratio > 0:
        # do ratio test, need second closest match
        for b in range(B):
            dist[b, :, idx1[b, :]] = float('inf')
        min1b, _ = torch.min(dist, dim=2)
        mask *= min1 / min1b < ratio

        for b in range(B):
            dist[b, :, idx1[b, :]] = min1[b, :]

    # [B, K1], [B, K1], [B, K1], [B, K1, K2]
    return idx1, min1, mask, dist


def error_metrics(yx1, yx2, matches, mask, dist, aflow, img2_w_h, success_px_limit, border=0):
    B, K1, K2 = dist.shape
    W2, H2 = img2_w_h
    active_area = (H2 - 2 * border) * (W2 - 2 * border)  # assumes that img1 is of same size (usually true)

    # ground truth image coords in img1 of detected features in img0
    gt_yx2 = -torch.ones_like(yx1)
    for b in range(B):
        # +0.5 for automatic rounding; nan => -9223372036854775808
        gt_yx2[b, :, :] = (aflow[b, :, yx1[b, 0, :], yx1[b, 1, :]] + 0.5).flip(dims=(0,)).long()

    # valid correspondence exists
    border = 0  # TODO: start using border when compatibility with old results is not needed anymore
    gt_mask = (border <= gt_yx2[:, 0, :]) * (border <= gt_yx2[:, 1, :]) * \
              (gt_yx2[:, 0, :] < H2 - border) * (gt_yx2[:, 1, :] < W2 - border)
    num_gt_matches = gt_mask.sum(dim=1)

    mask *= gt_mask
    num_matches = mask.sum(dim=1)

    acc = torch.zeros((B, 5), device=yx1.device)
    for b in range(B):
        if num_matches[b] == 0:
            acc[b, 0] = 0
            acc[b, 1] = 0
            acc[b, 2] = float('nan')
            acc[b, 3] = float('nan')
            acc[b, 4] = float('nan')
            continue

        # error distance in pixels
        err_dist_b = torch.norm((gt_yx2[b, :, mask[b, :]] - yx2[b, :, matches[b, mask[b, :]]]).float(), dim=0)

        # correct matches
        success_b = err_dist_b <= success_px_limit
        num_successes = success_b.sum()

        # features with gt detected per 100x100 px area
        acc[b, 0] = 1e4 * num_gt_matches[b] / active_area

        # success ratio (nan if no ground truth), aka matching score or M-score
        acc[b, 1] = num_successes / num_gt_matches[b]

        # inlier ratio (nan if no valid matches), aka mean matching accuracy or MMA
        acc[b, 2] = num_successes / num_matches[b]

        # average distance error of successful matches (nan if no inliers), aka Localization Error, LE
        acc[b, 3] = err_dist_b[success_b].mean()

        # calculate mean Average Precision (mAP)
        gt_dist_b = torch.sum(
            torch.pow((gt_yx2[b, :, mask[b, :]].view((2, num_matches[b], 1)).expand((2, num_matches[b], K2))
                       - yx2[b, :, :].view((2, 1, K2)).expand((2, num_matches[b], K2))
                       ).float(), 2), dim=0)
        labels = gt_dist_b < success_px_limit ** 2
        acc[b, 4] = mAP(dist[b, mask[b, :], :], labels, descending=False)

        if 0:
            from ..losses.ap import DifferentiableAP
            apfn = DifferentiableAP(bins=100, euclidean=False)
            map0 = acc[b, 4]
            map1 = apfn(1 - dist[b, mask[b, :], :], labels)
            map2 = apfn(1 - 0.5 * dist[b, mask[b, :], :], labels)
            print('mAP: %s vs %s vs %s' % (map0, *map(lambda t: t.nansum()/torch.logical_not(torch.isnan(t)).sum(), (map1, map2))))

    return acc


def mAP(scores, labels, descending=True):
    # dist.shape is (K1, K2)
    assert scores.shape == labels.shape, 'shapes of scores and labels does not match: %s, %s' % (scores.shape, labels.shape)

    index = torch.argsort(scores, dim=1, descending=descending)
    sorted_labels = torch.gather(labels, 1, index)

    # prepare for ap calculation
    cum_correct = sorted_labels.cumsum(dim=1)
    cum_precision = cum_correct / cum_correct[:, -1:]

    # average precision, per query
    ap = cum_precision.mean(dim=1)

    num_ap = torch.logical_not(torch.isnan(ap)).sum()
    return ap.nansum() / num_ap   # mAP


def load_model(path, device, model_only=False):
    model_type = 'orig' if os.path.basename(path)[:1] == '_' or path[-3:] == '.pt' else 'own'

    if model_type == 'orig':
        from navex.models.r2d2orig import R2D2
        model = R2D2(path=path)
        model.to(device)
    else:
        from navex.lightning.base import TrialWrapperBase
        model = TrialWrapperBase.load_from_checkpoint(path, map_location=device)
        model.trial.workers = 0
        model.trial.batch_size = 1
        model.use_gpu = model.on_gpu

        if model_only:
            model = model.trial.model

    return model


def is_rgb_model(model):
    found, fst, c, rgb = False, None, [model], None
    while not found:
        for fst in c:
            c = list(fst.children())
            if len(c) > 0:
                break
            elif hasattr(fst, 'in_channels'):
                found = True
                break

    assert found, 'could not find the expected channel width of model input'
    rgb = fst.in_channels == 3
    return rgb


def ordered_nested_dict(input):
    output = OrderedDict()
    for k, v in input.items():
        if isinstance(v, dict):
            output[k] = ordered_nested_dict(v)
        elif isinstance(v, set):
            assert False, 'sets not supported yet'
        else:
            output[k] = v
    return output


def reorder_cols(X, src_cols, trg_cols, defaults=None):
    defaults = defaults or {}
    Xn = np.stack([np.array(X)[:, src_cols.index(p)]
                   if p in src_cols
                   else np.ones((len(X),)) * defaults[p]
                   for p in trg_cols], axis=1)
    return Xn.tolist()


def gkern2d(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    w, h = (l[0], l[1]) if '__iter__' in dir(l) else (l, l)
    sx, sy = (sig[0], sig[1]) if '__iter__' in dir(sig) else (sig, sig)
    ax = np.arange(-w // 2 + 1., w // 2 + 1.)
    ay = np.arange(-h // 2 + 1., h // 2 + 1.)
    xx, yy = np.meshgrid(ax, ay)
    kernel = np.exp(-((xx / sx) ** 2 + (yy / sy) ** 2) / 2)
    return kernel / np.sum(kernel)


def bsphkern(l=5):
    """
    creates a binary spherical kernel
    """
    gkern = gkern2d(l=l, sig=l)
    limit = gkern[l // 2 if isinstance(l, int) else l[1] // 2, -1] * 0.995
    return np.array(gkern >= limit, dtype=np.uint8)


def asteroid_limb_mask(image, min_feature_intensity=50):
    import cv2

    _, mask = cv2.threshold(image, min_feature_intensity, 255, cv2.THRESH_BINARY)
    kernel = bsphkern(round(6 * image.shape[0] / 512) * 2 + 1)

    # exclude asteroid limb from feature detection
    mask = cv2.erode(mask, bsphkern(7), iterations=1)  # remove stars
    mask = cv2.dilate(mask, kernel, iterations=1)  # remove small shadows inside asteroid
    mask = cv2.erode(mask, kernel, iterations=2)  # remove asteroid limb

    return mask


def max_rect_bounded_by_quad_mask(mask: np.ndarray = None):
    assert len(mask.shape) == 2, 'wrong shape for 2d mask: %s' % (mask.shape,)
    n, m = mask.shape

    transposed = False
    if n > m:
        mask = mask.T
        n, m = mask.shape
        transposed = True

    scale = 1
    if n > 1000:
        scale = math.ceil(n / 1000)
        mask = mask[::scale, ::scale]
        n, m = mask.shape

    #  ...x0....x1.......
    #  .....*****........
    #  y0a.#######.y0b...
    #  ....#######*......
    #  ..**#######****...
    #  ..**#######*****..
    #  ...*#######******.
    #  y1a.#######******.
    #  .....************.
    #  ...x0.***x1*y1b*..
    #  ..................

    rects = np.zeros((n, 2, 4), dtype=np.float32)
    for j in range(n):
        idxs = np.where(mask[j, :])[0].astype(np.float32)
        if len(idxs) > 0:
            x0, x1 = idxs[0], idxs[-1]

            idxs = np.where(mask[:, round(x0)])[0].astype(np.float32)
            y0a, y1a = idxs[0], idxs[-1]

            idxs = np.where(mask[:, round(x1)])[0].astype(np.float32)
            y0b, y1b = idxs[0], idxs[-1]

            rects[j, 0, :] = x0, y0a, x1, y1a
            rects[j, 1, :] = x0, y0b, x1, y1b

    # calc intersection between all rects[:, 0] and all rects[:, 1]
    I = np.zeros((n, n, 4), dtype=np.float32)
    I[:, :, :2] = np.max(np.stack((
                            np.repeat(rects[:, 0:1, :2], n, axis=1),
                            np.repeat(np.swapaxes(rects[:, 1:2, :2], 0, 1), n, axis=0)
                        )), axis=0)
    I[:, :, 2:4] = np.min(np.stack((
                            np.repeat(rects[:, 0:1, 2:4], n, axis=1),
                            np.repeat(np.swapaxes(rects[:, 1:2, 2:4], 0, 1), n, axis=0)
                        )), axis=0)

    areas = np.clip(I[:, :, 2] - I[:, :, 0], 0, np.inf) * np.clip(I[:, :, 3] - I[:, :, 1], 0, np.inf)
    idx = np.argmax(areas)
    rect = np.round(I.reshape((-1, 4))[idx, :]).astype(np.int32)

    if transposed:
        rect = rect[(1, 0, 3, 2),]

    if scale > 1:
        rect = rect * scale + np.array([1, 1, -1, -1]) * (scale - 1)

    if False:
        j, i = np.unravel_index(idx, (n, n))

        import cv2

        if scale > 1:
            mask = cv2.resize(mask.astype(np.uint8), (m*scale, n*scale))

        if transposed:
            mask = mask.T

        n, m = mask.shape
        x0, y0, x1, y1 = rect
        img_arr = cv2.rectangle((mask.astype(np.uint8) * 128).reshape((n, m, 1)).repeat(3, axis=2),
                                (x0, y0), (x1, y1), color=[0, 255, 0], thickness=3)
        cv2.imshow('rect', cv2.resize(img_arr, (768, 768)))
        cv2.waitKey()

    return rect


def ap_to_latex():
    # need that https://github.com/HarisIqbal88/PlotNeuralNet is installed
    #  - have to do manually by cloning repo and placing to site-packages/plot_nn
    #  - need to copy the layers-folder to target latex repository
    from plot_nn.pycore import tikzeng as tz
    from plot_nn.pycore import blocks as bk

    arch = [
        tz.to_head(''),
        tz.to_cor(),
        tz.to_begin(),

        # INPUT
        tz.to_input('media/example-input.png'),

        # BACKBONE
        tz.to_ConvConvRelu(name='b1', s_filer='W', n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)",
                           height=40, depth=40, width=(2, 2)),
        tz.to_Pool(name="b1_pool", offset="(0,0,0)", to="(b1-east)",
                   height=32, depth=32, width=1, opacity=0.5),

        *bk.block_2ConvPool(name='b2', botton='b1_pool', top='b2_pool', s_filer='W/2', n_filer=64, offset="(1,0,0)",
                            size=(32, 32, 2), opacity=0.5),

        *bk.block_2ConvPool(name='b3', botton='b2_pool', top='b3_pool', s_filer='W/4', n_filer=128, offset="(1,0,0)",
                            size=(24, 24, 3.5), opacity=0.5),

        tz.to_ConvConvRelu(name='b4', s_filer='W/8', n_filer=(128, 128), offset="(1,0,0)", to='(b3_pool-east)',
                           height=18, depth=18, width=(3.5, 3.5)),
        tz.to_connection("b3_pool", "b4"),

        # DESCRIPTOR HEAD
        tz.to_ConvRelu(name='ba1', s_filer='', n_filer=256, offset="(2,7,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "ba1"),
        tz.to_Conv(name='ba2', s_filer='W/8', n_filer=256, offset="(0,0,0)", to='(ba1-east)',
                   height=18, depth=18, width=5.5),

        tz.to_SoftMax(name='ba3', s_filer='W/8', n_filer=256, offset="(1,0,0)", to='(ba2-east)',
                      height=18, depth=18, width=5.5, caption='descriptors'),
        tz.to_connection("ba2", "ba3"),

        # QUALITY HEAD
        tz.to_ConvRelu(name='bb1', s_filer='', n_filer=256, offset="(2,0,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "bb1"),
        tz.to_Conv(name='bb2', s_filer='W/8', n_filer=2, offset="(0,0,0)", to='(bb1-east)',
                   height=18, depth=18, width=1.5),

        tz.to_SoftMax(name='bb3', s_filer='W/8', n_filer=1, offset="(1,0,0)", to='(bb2-east)',
                      height=18, depth=18, width=1, caption='reliability'),
        tz.to_connection("bb2", "bb3"),

        # DETECTION HEAD
        tz.to_ConvRelu(name='bc1', s_filer='', n_filer=256, offset="(2,-8,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "bc1"),
        tz.to_Conv(name='bc2', s_filer='W/8', n_filer=65, offset="(0,0,0)", to='(bc1-east)',
                   height=18, depth=18, width=2),

        tz.to_SoftMax(name='bc3', s_filer='W', n_filer=1, offset="(2,0,0)", to='(bc2-east)',
                      height=40, depth=40, width=1, caption='repeatability'),
        tz.to_connection("bc2", "bc3"),

        tz.to_end()
    ]

    for c in arch:
        print(c)


def mob_to_latex():
    # need that https://github.com/HarisIqbal88/PlotNeuralNet is installed
    #  - have to do manually by cloning repo and placing to site-packages/plot_nn
    #  - need to copy the layers-folder to target latex repository
    from plot_nn.pycore import tikzeng as tz
    from plot_nn.pycore import blocks as bk

    arch = [
        tz.to_head(''),
        tz.to_cor(),
        tz.to_begin(),

        # INPUT
        tz.to_input('media/example-input.png'),

        # BACKBONE
        tz.to_ConvRelu(name='b1', s_filer='W/2', n_filer=16, offset="(0,0,0)", to="(0,0,0)",
                           height=32, depth=32, width=2),

        tz.to_ConvRelu(name='b2', s_filer='W/2', n_filer=16, offset="(1,0,0)", to="(b1-east)",
                           height=32, depth=32, width=2),
        tz.to_connection("b1", "b2"),

        tz.to_ConvRelu(name='b3', s_filer='W/4', n_filer=24, offset="(1,0,0)", to="(b2-east)",
                       height=24, depth=24, width=2.5),
        tz.to_connection("b2", "b3"),

        tz.to_ConvRelu(name='b4', s_filer='W/4', n_filer=24, offset="(1,0,0)", to="(b3-east)",
                       height=24, depth=24, width=2.5),
        tz.to_connection("b3", "b4"),

        tz.to_ConvRelu(name='b5', s_filer='W/8', n_filer=32, offset="(1,0,0)", to="(b4-east)",
                       height=18, depth=18, width=3),
        tz.to_connection("b4", "b5"),

        tz.to_ConvRelu(name='b6', s_filer='W/8', n_filer=64, offset="(1,0,0)", to="(b5-east)",
                       height=18, depth=18, width=4.5),
        tz.to_connection("b5", "b6"),

        tz.to_ConvRelu(name='b7', s_filer='W/8', n_filer=128, offset="(1,0,0)", to="(b6-east)",
                       height=18, depth=18, width=6),
        tz.to_connection("b6", "b7"),

        # DESCRIPTOR HEAD
        tz.to_ConvRelu(name='ba1', s_filer='W/8', n_filer=256, offset="(2,7,0)", to='(b7-east)',
                       height=18, depth=18, width=7.5),
        tz.to_connection("b7", "ba1"),

        tz.to_SoftMax(name='ba2', s_filer='W/8', n_filer=256, offset="(1,0,0)", to='(ba1-east)',
                      height=18, depth=18, width=7.5, caption='descriptors'),
        tz.to_connection("ba1", "ba2"),

        # QUALITY HEAD
        tz.to_ConvRelu(name='bb1', s_filer='W/8', n_filer=128, offset="(2,0,0)", to='(b7-east)',
                       height=18, depth=18, width=6),
        tz.to_connection("b7", "bb1"),

        tz.to_SoftMax(name='bb2', s_filer='W/8', n_filer=1, offset="(1,0,0)", to='(bb1-east)',
                      height=18, depth=18, width=1, caption='reliability'),
        tz.to_connection("bb1", "bb2"),

        # DETECTION HEAD
        tz.to_ConvRelu(name='bc1', s_filer='W/8', n_filer=128, offset="(2,-8,0)", to='(b7-east)',
                       height=18, depth=18, width=6),
        tz.to_connection("b7", "bc1"),

        tz.to_SoftMax(name='bc2', s_filer='W', n_filer=1, offset="(2,0,0)", to='(bc1-east)',
                      height=40, depth=40, width=1, caption='repeatability'),
        tz.to_connection("bc1", "bc2"),

        tz.to_end()
    ]

    for c in arch:
        print(c)


if __name__ == '__main__':
    ap_to_latex()

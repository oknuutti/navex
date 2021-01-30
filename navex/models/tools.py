import numpy as np

import torch
from torch import nn
from torch.functional import F


def detect_from_dense(des, det, qlt, top_k=None, det_lim=0.02, qlt_lim=0.02, interp='bicubic'):
    B, D, Hs, Ws = des.shape
    _, _, Ht, Wt = det.shape
    _, _, Hq, Wq = qlt.shape

    # interpolate if different scale heads
    if (Hs, Ws) != (Ht, Wt):
        des = F.interpolate(des, (Ht, Wt), mode=interp, align_corners=False)
    if (Hq, Wq) != (Ht, Wt):
        qlt = F.interpolate(qlt, (Ht, Wt), mode=interp, align_corners=False)

    # local maxima
    max_filter = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    maxima = (det == max_filter(det))  # [b, 1, h, w]

    # remove low confidence detections
    maxima *= (det >= det_lim)
    maxima *= (qlt >= qlt_lim)

    K = maxima.sum(dim=(2, 3)).max().item()
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
        descr[b, :, 0:k] = des[b, :, idxs[0], idxs[1]]

    return yx, scores, descr


def match(des1, des2, norm=2, mutual=True, ratio=False):
    B, D, K1 = des1.shape
    _, _, K2 = des2.shape
    dev = des1.device

    if K1 == 0 or K2 == 0:
        return torch.zeros((B, 2, K1), dtype=torch.long, device=dev), \
               torch.zeros((B, K1), dtype=torch.float, device=dev), \
               torch.zeros((B, K1), dtype=torch.bool, device=dev), \
               torch.zeros((B, K1, K2), dtype=torch.float, device=dev)

    dist = torch.linalg.norm(des1.view((B, D, K1, 1)).expand((B, D, K1, K2))
                             - des2.view((B, D, 1, K2)).expand((B, D, K1, K2)), ord=norm, dim=1)     # [b, k1, k2]
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

    return idx1, min1, mask, dist


def error_metrics(yx1, yx2, matches, mask, dist, aflow, img2_w_h, success_px_limit):
    B, K1, K2 = dist.shape
    W2, H2 = img2_w_h

    # ground truth image coords in img2 of detected features in img1
    gt_yx2 = -torch.ones_like(yx1)
    for b in range(B):
        # +0.5 for automatic rounding; nan => -9223372036854775808
        gt_yx2[b, :, :] = (aflow[b, :, yx1[b, 0, :], yx1[b, 1, :]] + 0.5).flip(dims=(0,)).long()

    # valid correspondence exists
    gt_mask = (0 <= gt_yx2[:, 0, :]) * (0 <= gt_yx2[:, 1, :]) * (gt_yx2[:, 0, :] < H2) * (gt_yx2[:, 1, :] < W2)
    num_true_matches = gt_mask.sum(dim=1)

    mask *= gt_mask
    num_matches = mask.sum(dim=1)

    acc = torch.zeros((B, 4), device=yx1.device)
    for b in range(B):
        if num_matches[b] == 0:
            acc[b, 0] = 0
            acc[b, 1] = float('nan')
            acc[b, 2] = float('nan')
            acc[b, 3] = float('nan')
            continue

        # error distance in pixels
        err_dist_b = torch.norm((gt_yx2[b, :, mask[b, :]] - yx2[b, :, matches[b, mask[b, :]]]).float(), dim=0)

        # correct matches
        success_b = err_dist_b < success_px_limit
        num_successes = success_b.sum()

        # success ratio (nan if no ground truth)
        acc[b, 0] = num_successes / num_true_matches[b]

        # inlier ratio (nan if no valid matches)
        acc[b, 1] = num_successes / num_matches[b]

        # average distance error of successful matches (nan if no inliers)
        acc[b, 2] = err_dist_b[success_b].mean()

        # calculate average precision
        x = 1 - 0.5 * dist[b, mask[b, :], :]
        gt_dist_b = torch.sum(torch.pow((gt_yx2[b, :, mask[b, :]].view((2, num_matches[b], 1)).expand((2, num_matches[b], K2))
                                         - yx2[b, :, :].view((2, 1, K2)).expand((2, num_matches[b], K2))
                                         ).float(), 2), dim=0)
        labels = gt_dist_b < success_px_limit ** 2

        from r2d2.nets.ap_loss import APLoss
        ap_loss = APLoss()
        ap_loss.to(x.device)
        t = ap_loss(x, labels)  # AP for each feature from img1
        num_ap = torch.logical_not(torch.isnan(t)).sum()
        acc[b, 3] = t.nansum() / num_ap  # mAP

    return acc


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
        tz.to_ConvConvRelu(name='b1', s_filer=512, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)",
                           height=40, depth=40, width=(2, 2)),
        tz.to_Pool(name="b1_pool", offset="(0,0,0)", to="(b1-east)",
                   height=32, depth=32, width=1, opacity=0.5),

        *bk.block_2ConvPool(name='b2', botton='b1_pool', top='b2_pool', s_filer=256, n_filer=64, offset="(1,0,0)",
                            size=(32, 32, 2), opacity=0.5),

        *bk.block_2ConvPool(name='b3', botton='b2_pool', top='b3_pool', s_filer=128, n_filer=128, offset="(1,0,0)",
                            size=(24, 24, 3.5), opacity=0.5),

        tz.to_ConvConvRelu(name='b4', s_filer=64, n_filer=(128, 128), offset="(1,0,0)", to='(b3_pool-east)',
                           height=18, depth=18, width=(3.5, 3.5)),
        tz.to_connection("b3_pool", "b4"),

        # DESCRIPTOR HEAD
        tz.to_ConvRelu(name='ba1', s_filer='', n_filer=256, offset="(2,7,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "ba1"),
        tz.to_Conv(name='ba2', s_filer=64, n_filer=256, offset="(0,0,0)", to='(ba1-east)',
                   height=18, depth=18, width=5.5),

        tz.to_SoftMax(name='ba3', s_filer=64, n_filer=256, offset="(1,0,0)", to='(ba2-east)',
                      height=18, depth=18, width=5.5, caption='descriptors'),
        tz.to_connection("ba2", "ba3"),

        # QUALITY HEAD
        tz.to_ConvRelu(name='bb1', s_filer='', n_filer=256, offset="(2,0,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "bb1"),
        tz.to_Conv(name='bb2', s_filer=64, n_filer=2, offset="(0,0,0)", to='(bb1-east)',
                   height=18, depth=18, width=1.5),

        tz.to_SoftMax(name='bb3', s_filer=64, n_filer=1, offset="(1,0,0)", to='(bb2-east)',
                      height=18, depth=18, width=1, caption='reliability'),
        tz.to_connection("bb2", "bb3"),

        # DETECTION HEAD
        tz.to_ConvRelu(name='bc1', s_filer='', n_filer=256, offset="(2,-8,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "bc1"),
        tz.to_Conv(name='bc2', s_filer=64, n_filer=65, offset="(0,0,0)", to='(bc1-east)',
                   height=18, depth=18, width=2),

        tz.to_SoftMax(name='bc3', s_filer=512, n_filer=1, offset="(2,0,0)", to='(bc2-east)',
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
        tz.to_ConvConvRelu(name='b1', s_filer=512, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)",
                           height=40, depth=40, width=(2, 2)),
        tz.to_Pool(name="b1_pool", offset="(0,0,0)", to="(b1-east)",
                   height=32, depth=32, width=1, opacity=0.5),

        *bk.block_2ConvPool(name='b2', botton='b1_pool', top='b2_pool', s_filer=256, n_filer=64, offset="(1,0,0)",
                            size=(32, 32, 2), opacity=0.5),

        *bk.block_2ConvPool(name='b3', botton='b2_pool', top='b3_pool', s_filer=128, n_filer=128, offset="(1,0,0)",
                            size=(24, 24, 3.5), opacity=0.5),

        tz.to_ConvConvRelu(name='b4', s_filer=64, n_filer=(128, 128), offset="(1,0,0)", to='(b3_pool-east)',
                           height=18, depth=18, width=(3.5, 3.5)),
        tz.to_connection("b3_pool", "b4"),

        # DESCRIPTOR HEAD
        tz.to_ConvRelu(name='ba1', s_filer='', n_filer=256, offset="(2,7,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "ba1"),
        tz.to_Conv(name='ba2', s_filer=64, n_filer=256, offset="(0,0,0)", to='(ba1-east)',
                   height=18, depth=18, width=5.5),

        tz.to_SoftMax(name='ba3', s_filer=64, n_filer=256, offset="(1,0,0)", to='(ba2-east)',
                      height=18, depth=18, width=5.5, caption='descriptors'),
        tz.to_connection("ba2", "ba3"),

        # QUALITY HEAD
        tz.to_ConvRelu(name='bb1', s_filer='', n_filer=128, offset="(2,0,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "bb1"),
        tz.to_Conv(name='bb2', s_filer=64, n_filer=1, offset="(0,0,0)", to='(bb1-east)',
                   height=18, depth=18, width=1.5),

        tz.to_SoftMax(name='bb3', s_filer=64, n_filer=1, offset="(1,0,0)", to='(bb2-east)',
                      height=18, depth=18, width=1, caption='reliability'),
        tz.to_connection("bb2", "bb3"),

        # DETECTION HEAD
        tz.to_ConvRelu(name='bc1', s_filer='', n_filer=128, offset="(2,-8,0)", to='(b4-east)',
                       height=18, depth=18, width=5.5),
        tz.to_connection("b4", "bc1"),
        tz.to_Conv(name='bc2', s_filer=64, n_filer=65, offset="(0,0,0)", to='(bc1-east)',
                   height=18, depth=18, width=2),

        tz.to_SoftMax(name='bc3', s_filer=512, n_filer=1, offset="(2,0,0)", to='(bc2-east)',
                      height=40, depth=40, width=1, caption='repeatability'),
        tz.to_connection("bc2", "bc3"),

        tz.to_end()
    ]

    for c in arch:
        print(c)


if __name__ == '__main__':
    mob_to_latex()

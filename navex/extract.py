
import argparse

import numpy as np
import torch
from pl_bolts.datamodules import AsynchronousLoader
from torch.functional import F
from torch.utils.data import DataLoader
from tqdm import tqdm

from navex.models.tools import is_rgb_model, load_model
from navex.datasets.base import ExtractionImageDataset
from navex.lightning.base import TrialWrapperBase
from navex.models import tools


# example:
#    python navex/extract.py --images data/hpatches/image_list_hpatches_sequences.txt \
#                            --model output/tune_s4c_44.ckpt \
#                            --tag ap4c --top-k 2000
from navex.models.r2d2orig import R2D2


def main():
    parser = argparse.ArgumentParser("extract features from images")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--images", type=str, required=True, help='images / list')
    parser.add_argument("--tag", type=str, required=True, help='output file tag')
    parser.add_argument("--top-k", type=int, default=None, help='limit on total number of keypoints')
    parser.add_argument("--feat-d", type=float, default=0.001, help='number of keypoints per pixel')
    parser.add_argument("--scale-f", type=float, default=2 ** (1/4))
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--det-lim", type=float, default=0.02)
    parser.add_argument("--qlt-lim", type=float, default=-10)
    parser.add_argument("--border", type=int, help="dont detect features if this close to image border")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    device = "cuda:0" if args.gpu else "cpu"
    model = load_model(args.model, device)
    rgb = is_rgb_model(model)
    model.eval()

    if args.border:
        border = args.border
    else:
        try:
            border = model.trial.loss_fn.border
        except:
            border = 16

    dataset = ExtractionImageDataset(args.images, rgb=rgb)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
    pbar = tqdm(data_loader)

    for i, data in enumerate(pbar):
        # if not ('v_abstract' in dataset.samples[i] or 'v_gardens' in dataset.samples[i]):
        #     continue
        data = data.to(device)

        # extract keypoints/descriptors for a single image
        cpu = False
        for _ in range(2):
            try:
                xys, desc, scores = extract_multiscale(model, data,
                                                       scale_f=args.scale_f,
                                                       min_scale=args.min_scale,
                                                       max_scale=args.max_scale,
                                                       min_size=args.min_size,
                                                       max_size=args.max_size,
                                                       top_k=args.top_k,
                                                       feat_d=args.feat_d,
                                                       det_lim=args.det_lim,
                                                       qlt_lim=args.qlt_lim,
                                                       border=border,
                                                       verbose=True)
                break
            except RuntimeError as e:
                # raise Exception('Problem with image #%d (%s)' % (i, dataset.samples[i])) from e
                print('Problem with image #%d (%s): %s' % (i, dataset.samples[i], str(e)))
                print('trying with CPU...')
                cpu = True
                model.cpu()
                data = data.cpu()
        if cpu:
            model.to(device)

        idxs = (-scores).argsort()
        if args.top_k is not None and args.top_k != 0:
            idxs = idxs[:args.top_k]

        outpath = dataset.samples[i] + '.' + args.tag
        with open(outpath, 'wb') as fh:
            np.savez(fh, imsize=data.shape[2:],
                     keypoints=xys[idxs],
                     descriptors=desc[idxs],
                     scores=scores[idxs])

        pbar.set_postfix({'scales': len(np.unique(xys[idxs, 2])), 'keypoints': len(idxs)})


def extract_multiscale(model, img0, scale_f=2 ** 0.25, min_scale=0.0, max_scale=1, min_size=256, max_size=1024,
                       top_k=None, feat_d=0.001, det_lim=None, qlt_lim=None, border=16, verbose=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    b, c, h0, w0 = img0.shape
    assert b == 1, "should be a batch with a single image"  # because can't fit different size images in same batch
    assert c in (1, 3), "should be an rgb or monochrome image"

    max_sc = min(max_scale, max_size / max(h0, w0))
    n = np.floor(np.log(max_sc) / np.log(scale_f))      # so that get one set of features at scale 1.0
    sc = min(max_sc, scale_f ** n)  # current scale factor

    XY, S, C, D = [], [], [], []
    while sc + 0.001 >= max(min_scale, min_size / max(h0, w0)):
        if sc - 0.001 <= min(max_scale, max_size / max(h0, w0)):
            if np.isclose(sc, 1, rtol=1e-2):
                img = img0
            else:
                # scale the image for next iteration
                h, w = round(h0 * sc), round(w0 * sc)
                mode = 'bilinear' if sc > 0.5 else 'area'
                img = F.interpolate(img0, (h, w), mode=mode, align_corners=False if mode == 'bilinear' else None)
            h, w = img.shape[2:]
            sc = w / w0

            # extract descriptors
            with torch.no_grad():
                des, det, qlt = model(img)

            if 0:
                import matplotlib.pyplot as plt
                from .visualize import plot_tensor
                fig, axs = plt.subplots(2, 1, figsize=(6, 6))
                plot_tensor(img, image=True, ax=axs[0])
                plot_tensor(qlt, ax=axs[1])
                plt.show()

            _, _, H1, W1 = det.shape
            yx, conf, descr = tools.detect_from_dense(des, det, qlt, top_k=top_k, feat_d=feat_d, det_lim=det_lim,
                                                      qlt_lim=qlt_lim, border=border)

            # accumulate multiple scales
            XY.append((yx[0].t().flip(dims=(1,)).float() / sc).cpu().numpy())
            S.append(((32 / sc) * torch.ones((len(descr[0, 0, :]), 1), dtype=torch.float32, device=des.device)).cpu().numpy())
            C.append(conf[0].t().cpu().numpy())
            D.append(descr[0].t().cpu().numpy())
        sc /= scale_f

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    XY = np.concatenate(XY)
    S = np.concatenate(S)  # scale
    XYS = np.concatenate([XY, S], axis=1)

    D = np.concatenate(D)
    C = np.concatenate(C).flatten()  # confidence

    return XYS, D, C


if __name__ == '__main__':
    main()


import argparse

import numpy as np
import torch
from torch.functional import F
from torch.utils.data import DataLoader
from tqdm import tqdm

from navex.models.tools import is_rgb_model, load_model
from .datasets.base import ExtractionImageDataset
from .lightning.base import TrialWrapperBase
from .models import tools


# example:
#    python navex/extract.py --images data/hpatches/image_list_hpatches_sequences.txt \
#                            --model output/tune_s4c_44.ckpt \
#                            --tag ap4c --top-k 2000
from navex.models.r2d2orig import R2D2


def main():
    parser = argparse.ArgumentParser("extract features from images")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--images", type=str, required=True, help='images / list')
    parser.add_argument("--tag", type=str, default='astr', help='output file tag')
    parser.add_argument("--top-k", type=int, default=2000, help='number of keypoints')
    parser.add_argument("--scale-f", type=float, default=2 ** (1/4))
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--min-scale", type=float, default=0)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument("--det-lim", type=float, default=0.02)
    parser.add_argument("--qlt-lim", type=float, default=-10)
    parser.add_argument("--border", type=int, default=16, help="dont detect features if this close to image border")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    device = "cuda:0" if args.gpu else "cpu"
    model = load_model(args.model, device)
    rgb = is_rgb_model(model)
    model.eval()

    dataset = ExtractionImageDataset(args.images, rgb=rgb)
    data_loader = model.wrap_ds(dataset)

    for i, data in enumerate(tqdm(data_loader)):
        data = data.to(device)

        # extract keypoints/descriptors for a single image
        xys, desc, scores = extract_multiscale(model, data,
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

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy().flatten()
        idxs = scores.argsort()[-args.top_k or None:]

        outpath = dataset.samples[i] + '.' + args.tag
        with open(outpath, 'wb') as fh:
            np.savez(fh, imsize=data.shape[2:],
                     keypoints=xys[idxs],
                     descriptors=desc[idxs],
                     scores=scores[idxs])


def extract_multiscale(model, img0, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1, min_size=256, max_size=1024,
                       top_k=None, det_lim=None, qlt_lim=None, border=16, verbose=False):
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

            if 1:
                import matplotlib.pyplot as plt
                from .visualize import plot_tensor
                fig, axs = plt.subplots(2, 1, figsize=(6, 6))
                plot_tensor(img, image=True, ax=axs[0])
                plot_tensor(qlt, ax=axs[1])
                plt.show()

            _, _, H1, W1 = det.shape
            yx, conf, descr = tools.detect_from_dense(des, det, qlt, top_k=top_k, det_lim=det_lim,
                                                      qlt_lim=qlt_lim, border=border)

            # accumulate multiple scales
            XY.append(yx[0].t().flip(dims=(1,)).float() / sc)
            S.append((32 / sc) * torch.ones((len(descr[0, 0, :]), 1), dtype=torch.float32, device=des.device))
            C.append(conf[0].t())
            D.append(descr[0].t())
        sc /= scale_f

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    XY = torch.cat(XY)
    S = torch.cat(S)  # scale
    XYS = torch.cat([XY, S], dim=1)

    D = torch.cat(D)
    C = torch.cat(C)  # confidence
    return XYS, D, C


if __name__ == '__main__':
    main()

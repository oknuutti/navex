import math
import os
import argparse

import numpy as np
import torch
from torch.functional import F
from torch.utils.data import DataLoader
from tqdm import tqdm

from navex.models.tools import is_rgb_model, load_model
from navex.datasets.base import ExtractionImageDataset, AugmentedPairDatasetMixin
from navex.models import tools
from navex.visualizations.misc import img2tensor


# example:
#    python navex/extract.py --images data/hpatches/image_list_hpatches_sequences.txt \
#                            --model output/tune_s4c_44.ckpt \
#                            --tag ap4c --top-k 2000


def main():
    def nullable_float(val):
        if val.strip().lower() in ('', 'none'):
            return None
        return float(val)

    parser = argparse.ArgumentParser("extract features from images")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--images", type=str, required=True, help='images / list')
    parser.add_argument("--recurse", type=int, default=1, help='find images in subfolders too')
    parser.add_argument("--tag", type=str, required=True, help='output file tag')
    parser.add_argument("--top-k", type=int, default=None, help='limit on total number of keypoints')
    parser.add_argument("--feat-d", type=nullable_float, default=0.001, help='number of keypoints per pixel')
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
    args = parser.parse_args()

    ext = Extractor(args.model, gpu=args.gpu, top_k=args.top_k, feat_d=args.feat_d, border=args.border,
                    scale_f=args.scale_f, min_size=args.min_size, max_size=args.max_size,
                    min_scale=args.min_scale, max_scale=args.max_scale, det_lim=args.det_lim,
                    qlt_lim=args.qlt_lim, mode=args.det_mode, kernel_size=args.kernel_size)

    sie = SingleImageExtractor(ext, save_ext=args.tag, verbose=True)

    sie.extract(args.images, args.recurse)


class SingleImageExtractor:
    def __init__(self, extractor: 'Extractor', save_ext=None, verbose=False):
        self.extractor = extractor
        self.save_ext = save_ext
        self.verbose = verbose

    def extract(self, images, recurse=True, debug_det=False):
        if self.save_ext is None:
            keypoint_arr = []
            descriptor_arr = []
            score_arr = []

        if not isinstance(images, DataLoader):
            dataset = ExtractionImageDataset(images, rgb=self.extractor.rgb, recurse=recurse)
            data_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
        else:
            data_loader = images

        pbar = tqdm(data_loader) if self.verbose else data_loader
        for i, data in enumerate(pbar):
            force_cpu = False
            for _ in range(2):
                try:
                    xys, desc, scores = self.extractor.extract(data, force_cpu=force_cpu, debug_det=debug_det)
                    break
                except RuntimeError as e:
                    # raise Exception('Problem with image #%d (%s)' % (i, dataset.samples[i])) from e
                    print('Problem with image #%d (%s): %s' % (i, data_loader.dataset.samples[i], str(e)))
                    if force_cpu or not self.extractor.gpu:
                        raise e
                    print('trying with CPU...')
                    force_cpu = True

            if self.save_ext is None:
                keypoint_arr.append(xys)
                descriptor_arr.append(desc)
                score_arr.append(scores)
            else:
                outpath = data_loader.dataset.samples[i] + '.' + self.save_ext
                with open(outpath, 'wb') as fh:
                    np.savez(fh, imsize=data.shape[2:],
                             keypoints=xys,
                             descriptors=desc,
                             scores=scores)
            if self.verbose:
                pbar.set_postfix({'scales': len(np.unique(xys[:, 2])), 'keypoints': len(xys)})

        if self.save_ext is None:
            if len(keypoint_arr) == 1:
                keypoint_arr, descriptor_arr, score_arr = keypoint_arr[0], descriptor_arr[0], score_arr[0]
            return keypoint_arr, descriptor_arr, score_arr


class Extractor:
    TRADITIONAL = {'akaze', 'surf', 'sift', 'rsift', 'orb'}

    def __init__(self, model, gpu=True, top_k=None, border=None, feat_d=0.001, scale_f=2 ** (1 / 4), min_size=256,
                 max_size=1024, min_scale=0.0, max_scale=1.0, det_lim=0.7, qlt_lim=0.7,
                 mode='nms', kernel_size=3, verbose=False):
        if gpu is None:
            gpu = torch.has_cuda
        if not gpu:
            torch.set_num_threads(os.cpu_count() // 2 - 1)

        self.gpu = gpu
        self.device = "cuda:0" if gpu else "cpu"

        if model not in self.TRADITIONAL:
            if model.endswith('.onnx'):
                import onnxruntime as ort
                self.model = ort.InferenceSession(model)
                self.rgb = '.rgb.' in model
            else:
                self.model = load_model(model, self.device)
                self.rgb = is_rgb_model(self.model)
                self.model.eval()
        else:
            self.model = model
            self.rgb = True

        if border is None:
            try:
                border = model.trial.loss_fn.border
            except:
                border = 16
        self.border = border

        self.top_k = top_k
        self.feat_d = feat_d
        self.scale_f = scale_f
        self.min_size = min_size
        self.max_size = max_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.det_lim = det_lim
        self.qlt_lim = qlt_lim
        self.mode = mode
        self.kernel_size = kernel_size
        self.verbose = verbose

    def extract(self, image, debug_det=False, force_cpu=False):
        if isinstance(self.model, str):
            if isinstance(image, torch.Tensor):
                from navex.visualizations.misc import tensor2img
                image = tensor2img(image)

            top_k = self.top_k
            if top_k is None:
                top_k = total_feature_count(image.shape[:2], self.scale_f, self.min_scale, self.max_scale,
                                            self.min_size, self.max_size, self.feat_d, self.border)

            xys, desc, scores = extract_traditional(self.model, image, max_size=self.max_size,
                                                    top_k=top_k, border=self.border)
        else:
            if not isinstance(image, torch.Tensor):
                image = img2tensor(image)

            if self.rgb and image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            elif not self.rgb and image.shape[1] == 3:
                image = image[:, 0:1, :, :]

            if isinstance(self.model, torch.nn.Module):
                if force_cpu:
                    image = image.cpu()
                    self.model.cpu()
                else:
                    image = image.to(self.device)
                    self.model.to(self.device)

            # extract keypoints/descriptors for a single image
            xys, desc, scores = extract_multiscale(self.model, image,
                                                   scale_f=self.scale_f,
                                                   min_scale=self.min_scale,
                                                   max_scale=self.max_scale,
                                                   min_size=self.min_size,
                                                   max_size=self.max_size,
                                                   top_k=self.top_k,
                                                   feat_d=self.feat_d,
                                                   det_lim=self.det_lim,
                                                   qlt_lim=self.qlt_lim,
                                                   det_mode=self.mode,
                                                   det_krn_size=self.kernel_size,
                                                   border=self.border,
                                                   verbose=self.verbose,
                                                   plot=debug_det)

        idxs = (-scores).argsort()
        if self.top_k is not None and self.top_k != 0:
            idxs = idxs[:self.top_k]

        return xys[idxs], desc[idxs], scores[idxs]


def total_feature_count(img_shape, scale_f=2 ** 0.25, min_scale=0.0, max_scale=1.0, min_size=256, max_size=1024,
                        feat_d=0.001, border=16):
    h0, w0 = img_shape
    max_sc = min(max_scale, max_size / max(h0, w0))
    n = np.floor(np.log(max_sc) / np.log(scale_f))      # so that get one set of features at scale 1.0
    sc = min(max_sc, scale_f ** n)  # current scale factor

    total = 0
    while sc + 0.001 >= max(min_scale, min_size / max(h0, w0)):
        if sc - 0.001 <= min(max_scale, max_size / max(h0, w0)):
            if np.isclose(sc, 1, rtol=1e-2):
                h, w = h0, w0
            else:
                h, w = round(h0 * sc), round(w0 * sc)
            sc = w / w0
            total += int((h - border * 2) * (w - border * 2) * feat_d)
        sc /= scale_f

    return total


def extract_multiscale(model, img0, scale_f=2 ** 0.25, min_scale=0.0, max_scale=1.0, min_size=256, max_size=1024,
                       top_k=None, feat_d=0.001, det_lim=None, qlt_lim=None, border=16,
                       det_mode='nms', det_krn_size=3, verbose=False, plot=False):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    b, c, h0, w0 = img0.shape
    assert b == 1, "should be a batch with a single image"  # because can't fit different size images in same batch
    assert c in (1, 3), "should be an rgb or monochrome image"

    max_sc = min(max_scale, max_size / max(h0, w0))
    n = np.floor(np.log(max_sc) / np.log(scale_f))      # so that get one set of features at scale 1.0
    sc = min(max_sc, scale_f ** n)  # current scale factor

    if sc + 0.001 < max(min_scale, min_size / max(h0, w0)) and max_size >= min_size and max_sc >= min_scale:
        sc = max_sc

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
            if isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    des, det, qlt = model(img)
            else:
                # is onnx model
                des, det, qlt = model.run(None, {'input': img.cpu().numpy()})
                des, det, qlt = torch.from_numpy(des), torch.from_numpy(det), torch.from_numpy(qlt)

            _, _, H1, W1 = det.shape
            yx, conf, descr = tools.detect_from_dense(des, det, qlt, top_k=top_k, feat_d=feat_d,
                                                      det_lim=det_lim, qlt_lim=qlt_lim, border=border,
                                                      kernel_size=det_krn_size, mode=det_mode)

            # accumulate multiple scales
            XY.append((yx[0].t().flip(dims=(1,)).float() / sc).cpu().numpy())
            S.append(((32 / sc) * torch.ones((len(descr[0, 0, :]), 1), dtype=torch.float32, device=des.device)).cpu().numpy())
            C.append(conf[0].t().cpu().numpy())
            D.append(descr[0].t().cpu().numpy())

            if plot:
                from .visualizations.misc import view_detections
                view_detections(img, det, qlt, title='Resolution: %dx%d, Scale: %.3f' % (w, h, sc))

        sc /= scale_f

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    if len(XY) == 0:
        print("WARNING: No features extracted! Maybe too demanding scaling constraints?")
        return np.empty((0, 3)), np.empty((0, 128)), np.empty((0,))

    XY = np.concatenate(XY)
    S = np.concatenate(S)  # scale
    XYS = np.concatenate([XY, S], axis=1)

    D = np.concatenate(D)
    C = np.concatenate(C).flatten()  # confidence

    return XYS, D, C


def extract_traditional(method, img, max_size=None, top_k=None, feat_d=None, border=16, asteroid_target=False):
    import cv2

    sc = min(1, max_size / max(img.shape[:2])) if max_size is not None else 1
    img = cv2.resize(img, None, fx=sc, fy=sc) if sc < 1 else img

    k = np.inf
    h, w = img.shape[:2]
    mask = tools.asteroid_limb_mask(img) if asteroid_target else None

    if feat_d is not None:
        # detect at most 0.001 features per pixel
        px_count = (h - border * 2) * (w - border * 2) if mask is None else np.sum(mask)
        k = min(k, int(px_count * feat_d))

    if top_k is not None:
        k = min(k, top_k)

    if method == 'orb':
        params = {
            'nfeatures': k,  # default: 500
            'edgeThreshold': 31,  # default: 31
            'fastThreshold': 10,  # default: 20
            'firstLevel': 0,  # always 0
            'nlevels': 8,  # default: 8
            'patchSize': 31,  # default: 31
            'scaleFactor': 1.189,  # default: 1.2
            'scoreType': cv2.ORB_HARRIS_SCORE,  # default ORB_HARRIS_SCORE, other: ORB_FAST_SCORE
            'WTA_K': 2,  # default: 2
        }
        det = cv2.ORB_create(**params)

    elif method == 'akaze':
        params = {
            'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,  # default: cv2.AKAZE_DESCRIPTOR_MLDB
            'descriptor_channels': 3,  # default: 3
            'descriptor_size': 0,  # default: 0
            'diffusivity': cv2.KAZE_DIFF_CHARBONNIER,  # default: cv2.KAZE_DIFF_PM_G2
            'threshold': 0.00005,  # default: 0.001
            'nOctaves': 4,  # default: 4
            'nOctaveLayers': 4,  # default: 4
        }
        det = cv2.AKAZE_create(**params)

    elif method in ('sift', 'rsift'):
        params = {
            'nfeatures': k,
            'nOctaveLayers': 4,  # default: 3
            'contrastThreshold': 0.01,  # default: 0.04
            'edgeThreshold': 10,  # default: 10
            'sigma': 1.6,  # default: 1.6
        }
        det = cv2.SIFT_create(**params)

    elif method == 'surf':
        params = {
            'hessianThreshold': 100.0,  # default: 100.0
            'nOctaves': 4,  # default: 4
            'nOctaveLayers': 4,  # default: 3
            'extended': False,  # default: False
            'upright': False,  # default: False
        }
        det = cv2.xfeatures2d.SURF_create(**params)
    else:
        assert False, 'invalid feature extraction method: %s' % method

    kps, dcs = det.detectAndCompute(img, mask)
    k = min(k, len(kps))
    idxs = (-np.array([kp.response for kp in kps])).argsort()
    idxs = idxs[:k]
    XYS = np.array([[kps[i].pt[0]/sc, kps[i].pt[1]/sc, kps[i].size] for i in idxs])
    D = np.array([dcs[i] for i in idxs])
    scores = np.array([kps[i].response for i in idxs])
    if method in ('orb', 'akaze'):
        D = D.astype(np.uint8)
    if method == 'rsift':
        D = np.sqrt(D / np.sum(D, axis=1, keepdims=True))

    return XYS, D, scores


if __name__ == '__main__':
    main()

import os
import re
import argparse

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torchvision.transforms import functional as F

from ..datasets.base import RGB_STD, RGB_MEAN, GRAY_STD, GRAY_MEAN
from ..models.tools import is_rgb_model, load_model
from .misc import plot_tensor

# TODO:
#   - show loss of each image as plot title
#   - plot x at the center of the image
#   - how big is the capturing area? plot a box on the images
#   - plot various cluster centers (cols) and variations of them (rows)
#   -
#   - plot detection and quality images
#   - try earlier model, the one that maximized the validation metric (maybe current model overfits?)
#   - try blurring image before feeding to model


def main():
    parser = argparse.ArgumentParser("Visualize features by generating prototypical images")
    parser.add_argument("--model", type=str, required=True, help='model path')
    parser.add_argument("--output", type=str, required=True, help='image output folder')
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--video", type=int, default=0)    # TODO: implement
    parser.add_argument("--cluster-center-file", help="load cluster centers from this file")
    parser.add_argument("--cluster-id", type=int, help="visualize descriptor corresponding to this cluster center")
    args = parser.parse_args()

    device = "cuda:0" if args.gpu else "cpu"
    model = load_model(args.model, device)
    rgb = is_rgb_model(model)
    model.eval()

    image_shape = (15, 3 if rgb else 1, args.height, args.width)
    loss_fn = nn.L1Loss() if 1 else nn.MSELoss()
    generator = RegularizedOutputSpecificImageGeneration(model, loss_fn, image_shape, device, orig_opt=False,
                                                         clipping_value=0.0, plot_freq=12,
                                                         anneal={100: 0.2, 150: 0.05},
                                                         periodic_blur=False, blur_sd=1.0, target_noise_sd=0.0,  # TODO: try with noise_sd
                                                         iters=200, lr=10e-0, wd=1e-3)

    cluster_centers, stats = np.load(args.cluster_center_file, allow_pickle=True)['arr_0']
    des_target = torch.Tensor(cluster_centers[args.cluster_id, :].reshape((1, -1, 1, 1))).to(device)
    det_target = qlt_target = torch.zeros((0,), device=device)
    target = (des_target, det_target, qlt_target)

    des_target_mask = torch.zeros((1, len(des_target), args.height, args.width), dtype=torch.bool, device=device)
    des_target_mask[0, :, args.height//2, args.width//2] = True
    det_target_mask = qlt_target_mask = torch.zeros((1, 1, args.height, args.width), dtype=torch.bool, device=device)
    target_mask = (des_target_mask, det_target_mask, qlt_target_mask)

    tensor_img = generator.generate(target, target_mask)
    img = to_image(tensor_img)

    os.makedirs(args.output, exist_ok=True)
    n = last_image_num(args.output) + 1
    cv2.imwrite(os.path.join(args.output, 'image_%04d.png' % n), rescale255(img))

    # plt.imshow(img)
    plt.show()


def last_image_num(path, pattern=r'^image_(\d+).(jpg|png)$'):
    max_id = -1
    for file in os.listdir(path):
        m = re.search(pattern, file)
        if m:
            max_id = max(max_id, int(m[1]))
    return max_id


def to_image(tensor_img, i=0):
    B, D, H, W = tensor_img.shape
    img = tensor_img[i, :, :, :].permute((1, 2, 0)).cpu().numpy()
    if D == 3:
        img = img * np.array(RGB_STD, dtype=np.float32) + np.array(RGB_MEAN, dtype=np.float32)
    else:
        img = img * np.array(GRAY_STD, dtype=np.float32) + np.array(GRAY_MEAN, dtype=np.float32)
    return img


def rescale255(img):
    v0, v1 = np.min(img), np.max(img)
    return np.clip((img - v0) * 255 / (v1 - v0), 0, 255).astype(np.uint8)


class RegularizedOutputSpecificImageGeneration:
    """
    Produces an image that maximizes a certain network output.
    Uses Gradient ascent, Gaussian blur, weight decay, and clipping.
    Based on
     - https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/generate_regularized_class_specific_samples.py
     - https://arxiv.org/pdf/1506.06579.pdf
     - own modifications as above works only for a single class of a classification network
    """

    def __init__(self, model, loss_fn, image_shape, device, init_sd=1.0, iters=200, blur_freq=4, blur_sd=1.0,
                 lr=1.0, wd=1e-4, clipping_value=0.1, plot_freq=25, target_noise_sd=0.01,
                 anneal=0.0, periodic_blur=True, orig_opt=False):
        """
        Besides the defaults, this combination has produced good images:
        blur_freq=6, blur_rad=0.8, wd = 0.05

        :param model:
        :param loss_fn: Loss function to use, e.g. torch.nn.L1Loss or torch.nn.MSELoss for descriptors,
                        torch.nn.BCELoss for detector or quality
        :param image_shape:
        :param device:
        :param init_sd:
        :param iters: Total iterations for gradient ascent (default: {150})
        :param blur_freq: Frequency of Gaussian blur effect, in iterations (default: {6})
        :param blur_sd: Gaussian blur sigma (default: {1.0})
        :param wd: Weight decay value for Stochastic Gradient Ascent (default: {0.05})
        :param clipping_value: Value for gradient clipping (default: {0.1})
        """
        assert blur_sd >= 0, 'blur rad is given in in pixels'

        self.model = model
        self.loss_fn = loss_fn
        self.image_shape = image_shape
        self.device = device
        self.init_sd = init_sd
        self.iters = iters
        self.blur_freq = blur_freq
        self.blur_sd = blur_sd
        self.lr = lr
        self.wd = wd
        self.clipping_value = clipping_value
        self.target_noise_sd = target_noise_sd
        self.plot_freq = plot_freq
        self.anneal = anneal
        self.periodic_blur = periodic_blur
        self.orig_opt = orig_opt

    def generate(self, target, target_mask):
        """
        Generate an image that

        :param target: target output values of targeted channels
        :param target_mask: output channels targeted
        :return: Generated image in tensor form
        """

        # Generate a random image
        if self.orig_opt:
            image = (torch.rand(self.image_shape, device=self.device) - GRAY_MEAN[0]) / GRAY_STD[0]
        else:
            image = self.init_sd * torch.randn(self.image_shape, device=self.device)
        image = Variable(image, requires_grad=True)

        self.model.eval()

        # Define optimizer for the image - use weight decay to add regularization
        # in SGD, wd = 2 * L2 regularization (https://bbabenko.github.io/weight-decay/)
        if not self.orig_opt:
            if 0:
                optimizer = SGD([image], lr=self.lr, momentum=0.9, dampening=0.0, weight_decay=self.wd)
            else:
                optimizer = Adam([image], lr=self.lr, betas=(0.9, 0.999), weight_decay=self.wd)

        ax = None
        pbar = tqdm(range(1, self.iters), desc="Optimizing Image")
        for i in pbar:
            # implement gaussian blurring every ith iteration to improve output
            if self.periodic_blur and self.blur_sd and i % self.blur_freq == 0:  # and i < self.iters/2:  # and False:
                with torch.no_grad():
                    image = F.gaussian_blur(image, [int(self.blur_sd * 3) * 2 + 1] * 2, [self.blur_sd] * 2)
                image = Variable(image, requires_grad=True)
                if not self.orig_opt:
                    optimizer.param_groups[0]['params'][0] = image

            if self.orig_opt:
                optimizer = Adam([image], lr=self.lr*1, weight_decay=self.wd*1)  # SGD or Adam

            # Forward
            if not self.periodic_blur and self.blur_sd:
                interm = F.gaussian_blur(image, [int(self.blur_sd * 3) * 2 + 1] * 2, [self.blur_sd] * 2)
            else:
                interm = image
            output = self.model(interm)

            # Massage outputs and targets
            if isinstance(output, (tuple, list)):
                assert isinstance(target, (tuple, list)) and isinstance(target_mask, (tuple, list)), \
                    'model output is of type tuple/list, target and target_mask must be matching length tuples'
                assert len(output) == len(target) == len(target_mask), \
                    'model output count does not correspond to target count'
                loss_fn = [self.loss_fn] * len(output) if not isinstance(self.loss_fn, (tuple, list)) else self.loss_fn
            else:
                output = [output]
                loss_fn = [self.loss_fn]
                if isinstance(target, (tuple, list)) and isinstance(target_mask, (tuple, list)):
                    assert len(target) == 1 and len(target_mask) == 1, \
                        'target and target_mask must be length 1 tuples/lists or tensors'
                else:
                    assert not isinstance(target, (tuple, list)) and not isinstance(target_mask, (tuple, list)), \
                        'model output is of type tensor, target and target_mask must be matching size tensors'
                    target = [target]
                    target_mask = [target_mask]

            loss = 0
            for out, trg, mask, lfn in zip(output, target, target_mask, loss_fn):
                selected_out = torch.reshape(torch.masked_select(out, mask), (-1, *trg.shape[1:]))
                noisy_trg = trg + self.target_noise_sd * torch.randn(trg.shape, device=self.device) \
                                if self.target_noise_sd > 0 else trg
                if noisy_trg.shape[0] != selected_out.shape[0]:
                    noisy_trg = noisy_trg.expand(selected_out.shape)
                loss = loss + torch.nan_to_num(lfn(selected_out, noisy_trg), 0)

            # Zero grads
            self.model.zero_grad()

            # Backward
            loss.backward()

            if self.clipping_value:
                torch.nn.utils.clip_grad_norm_(image, self.clipping_value)

            # Update image
            optimizer.step()

            # Anneal lr, wd
            if self.anneal and i in self.anneal:
                for pg in optimizer.param_groups:
                    pg['lr'] *= self.anneal[i]
                    pg['weight_decay'] *= self.anneal[i]

            pbar.set_postfix({'loss': "{0:.6f}".format(loss.item())})
            if self.plot_freq and i % self.plot_freq == 0:
                if ax is None:
                    bn = self.image_shape[0]
                    w = int(np.ceil(1.5 * np.sqrt(bn / 1.5)))
                    h = int(np.ceil(bn / w))
                    fig, ax = plt.subplots(h, w, sharex=True, sharey=True)
                    ax = ax.flatten() if isinstance(ax, np.ndarray) else ax
                with torch.no_grad():
                    plot_tensor(image, ax=ax)
                if 1:
                    plt.pause(0.05)
                else:
                    plt.show()
                    ax = None

        return image.detach()


if __name__ == '__main__':
    main()

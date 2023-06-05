import os
import abc
import math
from typing import Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tr

from .. import RND_SEED
from ..datasets.base import worker_init_fn
from ..datasets.tools import unit_aflow
from ..models import tools

from ..losses.r2d2 import R2D2Loss
from ..models.astropoint import AstroPoint
from ..models.disk import DISK
from ..models.hynet import HyNet
from ..models.mobile_ap import MobileAP
from ..models.r2d2 import R2D2


def _bare(val):
    if isinstance(val, Tensor):
        return val.item() if val.size() == 1 else val.detach().cpu().numpy()
    return val


class TrialBase(abc.ABC, torch.nn.Module):
    NAME = None  # override

    def __init__(self, model_conf, loss_conf, optimizer_conf, data_conf, batch_size, lr_scheduler=None,
                 aux_cost_coef=0.5, acc_grad_batches=1, hparams=None, accuracy_params=None):
        super(TrialBase, self).__init__()

        if isinstance(model_conf, dict):
            arch = model_conf['arch'].split('-')
            if len(arch) == 1:
                arch = 'ap'
            else:
                model_conf['arch'] = arch[1]
                arch = arch[0]

            # False is always a bad idea, leads to polarized qlt output
            model_conf['qlt_head']['single'] = True
            for k in ('partial_residual',):     # unused, TODO: remove from definition.yaml
                model_conf.pop(k)

            if isinstance(loss_conf, dict) and loss_conf['loss_type'] in ('disk', 'disk-p'):
                model_conf['qlt_head']['skip'] = True
                model_conf['train_with_raw_act_fn'] = loss_conf['loss_type'] == 'disk'
                if loss_conf['sampler']['max_neg_b'] < 0:
                    loss_conf['sampler']['max_neg_b'] = round(4 * (batch_size / 8) * (loss_conf['det_n'] / 8) ** 2)

            if arch == 'ap':
                model = AstroPoint(**model_conf)
            elif arch == 'r2d2':
                model_conf['des_head']['dimensions'] = 128
                model = R2D2(**model_conf)
            elif arch == 'disk':
                model = DISK(**model_conf)
            elif arch == 'hynet':
                model = HyNet(**model_conf)
            elif arch == 'mob':
                model = MobileAP(**model_conf)
            else:
                assert False, 'unknown main arch type "%s", valid ones are "ap" and "r2d2"' % arch
        else:
            model = model_conf

        self.model = model
        self.loss_fn = R2D2Loss(**loss_conf) if isinstance(loss_conf, dict) else loss_conf
        self.lr_scheduler = lr_scheduler
        self.aux_cost_coef = aux_cost_coef
        self.acc_grad_batches = acc_grad_batches
        self.loss_backward_fn = TrialBase._def_loss_backward_fn
        self.step_optimizer_fn = TrialBase._def_step_optimizer_fn
        self.optimizer = self.get_optimizer(**optimizer_conf) if isinstance(optimizer_conf, dict) else optimizer_conf
        self.hparams = {}

        # default values for accuracy calculations
        self.accuracy_params = dict(
            det_mode='nms', det_kernel_size=3, top_k=None, feat_d=0.001, border=16,
            mutual=True, ratio=False, success_px_limit=5, det_lim=0.5, qlt_lim=0.5
        )
        if accuracy_params is not None:
            self.accuracy_params.update(accuracy_params)

        try:
            from thop.profile import profile as ops_prof
            self.count_ops = ops_prof
        except:
            self.count_ops = False
        self.nparams = None
        self.macs = None

        self.target_macs = 20e9 / 256**2     # TODO: set at e.g. loss_conf
        self.data_conf = data_conf
        self.workers = int(os.getenv('CPUS', data_conf['workers']))
        self.batch_size = batch_size
        self.hparams = hparams or {
            'model': model_conf,
            'loss': loss_conf,
            'optimizer': optimizer_conf,
            'data_conf': data_conf,
            'batch_size': batch_size * acc_grad_batches,
        }

        self._tr_data, self._val_data, self._test_data = [None] * 3

    def update_conf(self, new_conf: dict, fail_silently: bool = False):
        for k, v in new_conf.items():
            ok = self.update_param(k, v)
            if not ok and not fail_silently:
                print("Cannot modify parameter `%s` after init" % (k,))

    def update_param(self, param, value):
        p, ok = param.split('.'), True
        if p[0] == 'loss':
            try:
                if len(p) > 1:
                    self.loss_fn.update_conf({'.'.join(p[1:]): value})
                else:
                    self.loss_fn.update_conf(value)
            except:
                ok = False
        elif p[0] == 'optimizer':
            pm = {'learning_rate': 'lr', 'weight_decay': 'weight_decay', 'eps': 'eps'}
            if len(p) > 1 and p[1] in pm:
                for pg in self.optimizer.param_groups:
                    pg[pm[p[1]]] = value
            elif len(p) == 1 and isinstance(value, dict):
                for pg in self.optimizer.param_groups:
                    for k, v in value.items():
                        if k in pm:
                            pg[pm[k]] = v
            else:
                ok = False
        elif p[0] == 'data':
            if p[1] in self.data_conf:
                self.data_conf[p[1]] = value
            else:
                ok = False
        else:
            ok = False
        return ok

    def get_optimizer(self, method, split_params, weight_decay, learning_rate, eps):
        # get optimizable params from both the network and the loss function
        params = [m.params_to_optimize(split=split_params)
                  for m in (self.model, self.loss_fn)]
        if split_params:
            params = [sum(p, []) for p in zip(*params)]
        else:
            params = sum(params, [])

        if split_params:
            assert method in ('adam', 'adabelief'), 'method not supported'
            new_biases, new_weights, biases, weights, others = params
            params = [
                {'params': new_biases, 'lr': learning_rate * 2, 'weight_decay': 0.0, 'eps': eps},
                {'params': new_weights, 'lr': learning_rate, 'weight_decay': weight_decay, 'eps': eps},
                {'params': biases, 'lr': learning_rate * 2, 'weight_decay': 0.0, 'eps': eps},
                {'params': weights, 'lr': learning_rate, 'weight_decay': weight_decay, 'eps': eps},
                {'params': others, 'lr': learning_rate, 'weight_decay': 0, 'eps': eps},
            ]

        if method == 'adam':
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, eps=eps)
        elif method == 'adabelief':
            from adabelief_pytorch import AdaBelief
            optimizer = AdaBelief(params, lr=learning_rate, weight_decay=weight_decay, eps=eps, weight_decouple=False,
                                  betas=(0.9, 0.999), rectify=False, fixed_decay=False, print_change_log=False)
        else:
            assert False, 'Invalid optimizer: %s' % method
        return optimizer

    @staticmethod
    def _def_loss_backward_fn(loss: Tensor):
        loss.backward()

    @staticmethod
    def _def_step_optimizer_fn(optimizer: Optimizer):
        optimizer.step()
        optimizer.zero_grad()

    def training_epoch_end(self, loss, tot, inl, dst, map):
        # called after each training epoch
        pass

    def train_batch(self, data: Tuple[Tensor, Tensor], labels: Tensor, epoch_idx: int, batch_idx: int,
                    component_loss=False, meta=None):

        # import matplotlib.pyplot as plt
        # import numpy as np
        # img0, img1 = data[0].permute((0, 2, 3, 1)).detach().numpy()[0,:,:,:], data[1].permute((0, 2, 3, 1)).detach().numpy()[0,:,:,:]
        # img0 = img0 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        # img1 = img1 * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        # plt.imshow(np.concatenate((img0, img1), axis=1))
        # plt.show()
        if 0:
            # enable to debug nans in gradient
            torch.autograd.set_detect_anomaly(True)

        self.model.train()
        output1 = self.model(data[0])
        output2 = self.model(data[1])

        if self.model.aux_qty:
            loss = self.loss(output1[0], output2[0], labels)
            for i in range(1, len(output1)):
                loss = loss + self.aux_cost_coef * self.loss(output1[i], output2[i], labels)
            output1, output2 = output1[0], output2[0]
        else:
            loss = self.loss(output1, output2, labels, component_loss=component_loss)

        self.loss_backward_fn(loss.sum(dim=1))

        if (batch_idx+1) % self.acc_grad_batches == 0:
            self.step_optimizer_fn(self.optimizer)

        with torch.no_grad():
            if self.model.conf.get('train_with_raw_act_fn', False):
                # for R2D2-DISK experiment, remove if unsuccessful and not used in next experiment
                (des1, det1, qlt1), (des2, det2, qlt2) = output1, output2
                det1 = self.model.activation(det1, fn_type=self.model.conf['det_head']['act_fn_type'])
                qlt1 = self.model.activation(qlt1, fn_type=self.model.conf['qlt_head']['act_fn_type'])
                det2 = self.model.activation(det2, fn_type=self.model.conf['det_head']['act_fn_type'])
                qlt2 = self.model.activation(qlt2, fn_type=self.model.conf['qlt_head']['act_fn_type'])
                output1_, output2_ = (des1, det1, qlt1), (des2, det2, qlt2)

            acc = self.accuracy(output1, output2, labels)   # TODO: use "meta" datastruct for pose estim accuracy

        return loss, acc

    def evaluate_batch(self, data: Tuple[Tensor, Tensor], labels: Tensor, component_loss=False):
        self.model.eval()
        with torch.no_grad():
            if self.count_ops:
                d = data[0][:1, :, :, :]
                macs, self.nparams = self.count_ops(self.model, inputs=(d,), verbose=False)
                self.macs = macs / d.shape[-1] / d.shape[-2]
                print('Params: %.2fM, MAC ops: %.2fG (with input dims: %s), MAC ops per px: %.1fk/px'
                      % (self.nparams * 1e-6, macs * 1e-9, d.shape, self.macs*1e-3))
                self.count_ops = False

            output1 = self.model(data[0])
            output2 = self.model(data[1])
            if self.model.aux_qty and self.model.training:
                output1, output2 = output1[0], output2[0]
            validation_loss = self.loss(output1, output2, labels, component_loss=component_loss)
            accuracy = self.accuracy(output1, output2, labels)
        return validation_loss, accuracy, (output1, output2)

    def loss(self, output1: Tensor, output2: Tensor, labels: Tensor, component_loss=False):
        assert self.loss_fn is not None, 'loss function not implemented'
        return self.loss_fn(output1, output2, labels, component_loss=component_loss)

    def accuracy(self, output1: Tensor, output2: Tensor, aflow: Tensor):

        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2
        _, _, H1, W1 = det1.shape
        _, _, H2, W2 = det2.shape
        p = self.accuracy_params

        yx1, conf1, descr1 = tools.detect_from_dense(des1, det1, qlt1, top_k=p['top_k'], feat_d=p['feat_d'],
                                                     det_lim=p['det_lim'], qlt_lim=p['qlt_lim'], border=p['border'],
                                                     mode=p['det_mode'], kernel_size=p['det_kernel_size'])
        yx2, conf2, descr2 = tools.detect_from_dense(des2, det2, qlt2, top_k=p['top_k'], feat_d=p['feat_d'],
                                                     det_lim=p['det_lim'], qlt_lim=p['qlt_lim'], border=p['border'],
                                                     mode=p['det_mode'], kernel_size=p['det_kernel_size'])

        # [B, K1], [B, K1], [B, K1], [B, K1, K2]
        matches, norm, mask, dist = tools.match(descr1, descr2, mutual=p['mutual'], ratio=p['ratio'])

        return tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), p['success_px_limit'],
                                   border=p['border'])

    def on_train_batch_end(self, losses, accuracies, accumulating_grad: bool):
        if hasattr(self.loss_fn, 'batch_end_update') and not accumulating_grad:
            num_val = torch.nansum(torch.logical_not(torch.isnan(accuracies)), dim=0)
            accs = torch.Tensor([float('nan')] * num_val.numel()).to(accuracies.device)
            if torch.sum(num_val > 0) > 0:
                accs[num_val > 0] = torch.nansum(accuracies[:, num_val > 0], dim=0) / num_val[num_val > 0]
            self.loss_fn.batch_end_update(accs)

    def log_values(self):
        log = {}
        funs = {'n': lambda x: x, 'e': lambda x: torch.exp(-x)}
        for p, f in (('wdt', 'e'), ('wap', 'e'), ('wqt', 'e'), ('base', 'n'), ('ap_base', 'n'), ('wpk', 'n')):
            val = getattr(self.loss_fn, p, None)
            if isinstance(val, torch.Tensor):
                log[p] = funs[f](val)
        return log or None

    def resource_loss(self, loss):
        # use self.macs and self.target_macs, something like this: loss * some_good_fn(self.macs, self.target_macs)
        if self.target_macs is not None and self.macs is not None:
            return loss + 2 * math.log(max(1, self.macs / self.target_macs))
        else:
            return loss

    def wrap_ds(self, dataset, shuffle=False):
        generator = None
        if shuffle:
            # second batch already differs significantly, not sure how to solve, better just use shuffle=False
            generator = torch.Generator()
            generator.manual_seed(RND_SEED)
        dl = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers,
                        shuffle=shuffle, generator=generator, pin_memory=True, worker_init_fn=worker_init_fn)
        return dl

    def build_training_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[0]

    def build_validation_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[1]

    def build_test_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[2]

    @abc.abstractmethod
    def _get_datasets(self, rgb):
        raise NotImplementedError()


class StudentTrialMixin:
    def __init__(self, teacher):
        self.teacher = teacher
        self.teacher.eval()

    def train_batch(self, data: Tuple[Tensor, Tensor], epoch_idx: int, batch_idx: int, component_loss: bool = False):
        clean_data, noisy_data = data

        with torch.no_grad():
            labels = self.teacher(clean_data)

        self.model.train()
        output = self.model(noisy_data)
        loss = self.loss(output, labels, component_loss=component_loss)
        self.loss_backward_fn(loss.sum(dim=1))

        if (batch_idx+1) % self.acc_grad_batches == 0:
            self.step_optimizer_fn(self.optimizer)

        with torch.no_grad():
            acc = self.accuracy(output, labels)

        return loss, acc

    def evaluate_batch(self, data: Tuple[Tensor, Tensor], component_loss: bool = False):
        assert isinstance(data, (tuple, list)), 'data must be a tuple or a list of (clean, noisy) tensors'
        clean_data, noisy_data = data

        self.model.eval()
        with torch.no_grad():
            if self.count_ops:
                d = clean_data[:1, :, :, :]
                macs, self.nparams = self.count_ops(self.model, inputs=(d,), verbose=False)
                self.macs = macs / d.shape[-1] / d.shape[-2]
                print('Params: %.2fM, MAC ops: %.2fG (with input dims: %s), MAC ops per px: %.1fk/px'
                      % (self.nparams * 1e-6, macs * 1e-9, d.shape, self.macs*1e-3))
                self.count_ops = False

            labels = self.teacher(clean_data)
            output = self.model(noisy_data)
            validation_loss = self.loss(output, labels, component_loss=component_loss)
            accuracy = self.accuracy(output, labels)
        return validation_loss, accuracy, (output, labels)

    def loss(self, output: Tensor, labels: Tensor, component_loss=False):
        assert self.loss_fn is not None, 'loss function not implemented'
        return self.loss_fn(output, labels, component_loss=component_loss)

    def accuracy(self, output: Tensor, labels: Tensor):
        des1, det1, qlt1 = output
        des2, det2, qlt2 = labels
        B, _, H1, W1 = det1.shape
        _, _, H2, W2 = det2.shape
        p = self.accuracy_params.copy()

        # skipped_qlt = self.model.conf.get('qlt_head', {'skip': False}).get('skip', False)
        # if skipped_qlt:
        #     p['det_lim'] *= 0.5

        yx1, conf1, descr1 = tools.detect_from_dense(des1, det1, qlt1, top_k=p['top_k'], feat_d=p['feat_d'],
                                                     det_lim=p['det_lim'], qlt_lim=p['qlt_lim'], border=p['border'],
                                                     mode=p['det_mode'], kernel_size=p['det_kernel_size'])
        yx2, conf2, descr2 = tools.detect_from_dense(des2, det2, qlt2, top_k=p['top_k'], feat_d=p['feat_d'],
                                                     det_lim=p['det_lim'], qlt_lim=p['qlt_lim'], border=p['border'],
                                                     mode=p['det_mode'], kernel_size=p['det_kernel_size'])

        # [B, K1], [B, K1], [B, K1], [B, K1, K2]
        matches, norm, mask, dist = tools.match(descr1, descr2, mutual=p['mutual'], ratio=p['ratio'])

        aflow = tr.ToTensor()(unit_aflow(W2, H2)).expand((B, 2, W2, H2)).to(yx1.device)
        return tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), p['success_px_limit'],
                                   border=p['border'])

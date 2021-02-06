import abc
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.functional import F
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tr

from .. import RND_SEED
from ..datasets.base import worker_init_fn, unit_aflow
from ..models import tools


def _bare(val):
    if isinstance(val, Tensor):
        return val.item() if val.size() == 1 else val.detach().cpu().numpy()
    return val


class TrialBase(abc.ABC, torch.nn.Module):
    def __init__(self, model, loss_fn, optimizer_conf, lr_scheduler=None, aux_cost_coef=0.5, acc_grad_batches=1):
        super(TrialBase, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.aux_cost_coef = aux_cost_coef
        self.acc_grad_batches = acc_grad_batches
        self.loss_backward_fn = TrialBase._def_loss_backward_fn
        self.step_optimizer_fn = TrialBase._def_step_optimizer_fn
        self.optimizer = self.get_optimizer(**optimizer_conf) if isinstance(optimizer_conf, dict) else optimizer_conf
        self.hparams = {}

        try:
            from thop.profile import profile as ops_prof
            self.count_ops = ops_prof
        except:
            self.count_ops = False

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

    def train_batch(self, data: Tuple[Tensor, Tensor], labels: Tensor, epoch_idx: int, batch_idx: int):
        self.model.train()
        output1 = self.model(data[0])
        output2 = self.model(data[1])

        if self.model.aux_qty:
            loss = self.loss(output1[0], output2[0], labels)
            for i in range(1, len(output1)):
                loss = loss + self.aux_cost_coef * self.loss(output1[i], output2[i], labels)
            output1, output2 = output1[0], output2[0]
        else:
            loss = self.loss(output1, output2, labels)

        self.loss_backward_fn(loss)

        if (batch_idx+1) % self.acc_grad_batches == 0:
            self.step_optimizer_fn(self.optimizer)

        return loss, (output1, output2)

    def evaluate_batch(self, data: Tuple[Tensor, Tensor], labels: Tensor, **acc_conf):
        self.model.eval()
        with torch.no_grad():
            if self.count_ops:
                d = data[0][:1, :, :, :]
                macs, params = self.count_ops(self.model, inputs=(d,), verbose=False)
                print('Params: %.2fM, MAC ops: %.2fG (with input dims: %s)' % (params * 1e-6, macs * 1e-9, d.shape))
                self.count_ops = False

            output1 = self.model(data[0])
            output2 = self.model(data[1])
            if self.model.aux_qty and self.model.training:
                output1, output2 = output1[0], output2[0]
            validation_loss = self.loss(output1, output2, labels)
            accuracy = self.accuracy(output1, output2, labels, **acc_conf)
        return validation_loss, accuracy, (output1, output2)

    def loss(self, output1: Tensor, output2: Tensor, labels: Tensor):
        assert self.loss_fn is not None, 'loss function not implemented'
        return self.loss_fn(output1, output2, labels)

    def accuracy(self, output1: Tensor, output2: Tensor, aflow: Tensor, top_k=None, border=16,
                 mutual=True, ratio=False, success_px_limit=6, det_lim=0.02, qlt_lim=0.02):

        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2
        _, _, H2, W2 = det2.shape

        if top_k is None:
            # detect at most 0.002 features per pixel
            top_k = int((H2 - border * 2) * (W2 - border * 2) * 0.002)

        yx1, conf1, descr1 = tools.detect_from_dense(des1, det1, qlt1, top_k=top_k, det_lim=det_lim,
                                                     qlt_lim=qlt_lim, border=border)
        yx2, conf2, descr2 = tools.detect_from_dense(des2, det2, qlt2, top_k=top_k, det_lim=det_lim,
                                                     qlt_lim=qlt_lim, border=border)

        # [B, K1], [B, K1], [B, K1], [B, K1, K2]
        matches, norm, mask, dist = tools.match(descr1, descr2, mutual=mutual, ratio=ratio)

        return tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), success_px_limit)

    def log_values(self):
        """
        override to return parameters to be logged during training, validation and testing
        :return: dict with param:value pairs to be logged
        """
        return None

    @abc.abstractmethod
    def build_training_data_loader(self, rgb=False):
        raise NotImplemented()

    @abc.abstractmethod
    def build_validation_data_loader(self, rgb=False):
        raise NotImplemented()

    @abc.abstractmethod
    def build_test_data_loader(self, rgb=False):
        raise NotImplemented()

    def wrap_ds(self, dataset, shuffle=False):
        generator = None
        if shuffle:
            # second batch already differs significantly, not sure how to solve, better just use shuffle=False
            generator = torch.Generator()
            generator.manual_seed(RND_SEED)
        dl = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.workers,
                        shuffle=shuffle, generator=generator, pin_memory=True, worker_init_fn=worker_init_fn)
        return dl


class StudentTrialMixin:
    def __init__(self, teacher):
        self.teacher = teacher
        self.teacher.eval()

    def train_batch(self, data: Tensor, epoch_idx: int, batch_idx: int):
        with torch.no_grad():
            labels = self.teacher(data)

        self.model.train()
        output = self.model(data)
        loss = self.loss(output, labels)
        self.loss_backward_fn(loss)

        if (batch_idx+1) % self.acc_grad_batches == 0:
            self.step_optimizer_fn(self.optimizer)

        return loss, (output, labels)

    def evaluate_batch(self, data: Tensor, **acc_conf):
        self.model.eval()
        with torch.no_grad():
            if self.count_ops:
                d = data[:1, :, :, :]
                macs, params = self.count_ops(self.model, inputs=(d,), verbose=False)
                print('Params: %.2fM, MAC ops: %.2fG (with input dims: %s)' % (params * 1e-6, macs * 1e-9, d.shape))
                self.count_ops = False

            output = self.model(data)
            labels = self.teacher(data)
            validation_loss = self.loss(output, labels)
            accuracy = self.accuracy(output, labels, **acc_conf)
        return validation_loss, accuracy, (output, labels)

    def loss(self, output: Tensor, labels: Tensor):
        assert self.loss_fn is not None, 'loss function not implemented'
        return self.loss_fn(output, labels)

    def accuracy(self, output: Tensor, labels: Tensor, top_k=None, border=16,
                 mutual=True, ratio=False, success_px_limit=6, det_lim=0.02, qlt_lim=0.02):

        des1, det1, qlt1 = output
        des2, det2, qlt2 = labels
        B, _, H2, W2 = det2.shape

        if top_k is None:
            # detect at most 0.002 features per pixel
            top_k = int((H2 - border * 2) * (W2 - border * 2) * 0.002)

        yx1, conf1, descr1 = tools.detect_from_dense(des1, det1, qlt1, top_k=top_k, det_lim=det_lim,
                                                     qlt_lim=qlt_lim, border=border)
        yx2, conf2, descr2 = tools.detect_from_dense(des2, det2, qlt2, top_k=top_k, det_lim=det_lim,
                                                     qlt_lim=qlt_lim, border=border)

        # [B, K1], [B, K1], [B, K1], [B, K1, K2]
        matches, norm, mask, dist = tools.match(descr1, descr2, mutual=mutual, ratio=ratio)

        aflow = tr.ToTensor()(unit_aflow(W2, H2)).expand((B, 2, W2, H2))
        return tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), success_px_limit)

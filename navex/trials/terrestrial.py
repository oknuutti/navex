import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

from ..datasets.aachen import AachenFlowDataset
from ..datasets.base import worker_init_fn
from ..losses.r2d2 import R2D2Loss
from ..models.astropoint import AstroPoint
from .base import TrialBase


class TerrestrialTrial(TrialBase):
    def __init__(self, model_conf, loss_conf, optimizer_conf, data_conf, batch_size, acc_grad_batches=1, hparams=None):
        super(TerrestrialTrial, self).__init__(
            model=AstroPoint(**model_conf) if isinstance(model_conf, dict) else model_conf,
            loss_fn=R2D2Loss(**loss_conf) if isinstance(loss_conf, dict) else loss_conf,
            optimizer_conf=optimizer_conf,
            acc_grad_batches=acc_grad_batches)

        self.data_conf = data_conf
        self.batch_size = batch_size
        self.hparams = hparams or {
            'model': model_conf,
            'loss': loss_conf,
            'optimizer': optimizer_conf,
            'data_conf': data_conf,
            'batch_size': batch_size * acc_grad_batches,
        }

        self._tr_data, self._val_data, self._test_data = [None] * 3

    def get_optimizer(self, method, split_params, excl_bn, weight_decay, learning_rate, eps):
        # get optimizable params from both the network and the loss function
        params = [m.params_to_optimize(split=split_params, excl_batch_norm=excl_bn)
                  for m in (self.model, self.loss_fn)]
        if split_params:
            params = [sum(p, []) for p in zip(*params)]
        else:
            params = sum(params, [])

        if method == 'adam':
            if split_params:
                new_biases, new_weights, biases, weights, others = params
                optimizer = torch.optim.Adam([
                    {'params': new_biases, 'lr': learning_rate * 2, 'weight_decay': 0.0, 'eps': eps},
                    {'params': new_weights, 'lr': learning_rate, 'weight_decay': weight_decay, 'eps': eps},
                    {'params': biases, 'lr': learning_rate * 2, 'weight_decay': 0.0, 'eps': eps},
                    {'params': weights, 'lr': learning_rate, 'weight_decay': weight_decay, 'eps': eps},
                    {'params': others, 'lr': learning_rate, 'weight_decay': 0, 'eps': eps},
                ])
            else:
                optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, eps=eps)
        else:
            assert False, 'Invalid optimizer: %s' % method
        return optimizer

    def build_training_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[0]

    def build_validation_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[1]

    def build_test_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[2]

    def _get_datasets(self, rgb):
        if self._tr_data is None:
            fullset = AachenFlowDataset(self.data_conf['path'], eval=False, rgb=rgb)
            datasets = fullset.split(self.data_conf.get('trn_ratio', 0.8),
                                     self.data_conf.get('val_ratio', 0.1),
                                     self.data_conf.get('tst_ratio', 0.1), eval=(2,), rgb=rgb)
            self._tr_data, self._val_data, self._test_data = \
                    self._wrap_ds(datasets[0], shuffle=True), self._wrap_ds(datasets[1]), self._wrap_ds(datasets[2])
        return self._tr_data, self._val_data, self._test_data

    def _wrap_ds(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_conf['workers'],
                          shuffle=shuffle, pin_memory=True, worker_init_fn=worker_init_fn)


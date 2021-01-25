import json
import math
import os

import torch
from torch.utils.data import ConcatDataset

from ..datasets.aachen import AachenFlowDataset, AachenSynthPairDataset, AachenStyleTransferDataset
from ..datasets.base import AugmentedConcatDataset
from ..datasets.revisitop1m import WebImageSynthPairDataset
from ..losses.r2d2 import R2D2Loss
from ..models.astropoint import AstroPoint
from .base import TrialBase
from ..models.r2d2 import R2D2


class TerrestrialTrial(TrialBase):
    def __init__(self, model_conf, loss_conf, optimizer_conf, data_conf, batch_size, acc_grad_batches=1, hparams=None):
        if isinstance(model_conf, dict):
            arch = model_conf['arch'].split('-')
            if len(arch) == 1:
                arch = 'ap'
            else:
                model_conf['arch'] = arch[1]
                arch = arch[0]

            if arch == 'ap':
                model = AstroPoint(**model_conf)
            elif arch == 'r2d2':
                for k in ('head_conv_ch', 'direct_detection', 'dropout'):
                    model_conf.pop(k)
                model_conf['descriptor_dim'] = 128
                model = R2D2(**model_conf)
            else:
                assert False, 'unknown main arch type "%s", valid ones are "ap" and "r2d2"' % arch
        else:
            model = model_conf

        super(TerrestrialTrial, self).__init__(
            model=model,
            loss_fn=R2D2Loss(**loss_conf) if isinstance(loss_conf, dict) else loss_conf,
            optimizer_conf=optimizer_conf,
            acc_grad_batches=acc_grad_batches)

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

    def log_values(self):
        log = {}
        if not isinstance(self.loss_fn.wdt, float):
            log['wdt'] = self.loss_fn.wdt
        if not isinstance(self.loss_fn.wap, float):
            log['wap'] = self.loss_fn.wap
        return log or None

    def build_training_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[0]

    def build_validation_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[1]

    def build_test_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[2]

    def _get_datasets(self, rgb):
        if self._tr_data is None:
            npy = json.loads(self.data_conf['npy'])
            common = dict(eval=False, rgb=rgb, npy=npy)
            dconf = {k: v for k, v in self.data_conf.items() if k in ('noise_max', 'rnd_gain', 'image_size')}
            sconf = dict(max_tr=0, max_rot=math.radians(12), max_shear=0.2, max_proj=0.7)

            ds = []
            if 1:
                ds.append(AachenFlowDataset(self.data_conf['path'], **common, **dconf))
            if 1:
                ds.append(WebImageSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                ds.append(AachenStyleTransferDataset(self.data_conf['path'], **common, **dconf))
            if 1:
                ds.append(AachenSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))

            fullset = AugmentedConcatDataset(ds)
            datasets = fullset.split(self.data_conf.get('trn_ratio', 0.8),
                                     self.data_conf.get('val_ratio', 0.1),
                                     self.data_conf.get('tst_ratio', 0.1), eval=(2,))

            self._tr_data, self._val_data, self._test_data = \
                self.wrap_ds(datasets[0]), self.wrap_ds(datasets[1]), self.wrap_ds(datasets[2])
        return self._tr_data, self._val_data, self._test_data

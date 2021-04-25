import json
import math
import os

import torch

from navex.datasets.terrestrial.aachen import AachenFlowPairDataset, AachenSynthPairDataset, AachenStyleTransferPairDataset
from ..datasets.base import AugmentedConcatDataset
from navex.datasets.terrestrial.revisitop1m import WebImageSynthPairDataset
from ..losses.r2d2 import R2D2Loss
from ..models.astropoint import AstroPoint
from .base import TrialBase
from ..models.mobile_ap import MobileAP
from ..models.r2d2 import R2D2


class TerrestrialTrial(TrialBase):
    NAME = 'terr'

    def __init__(self, model_conf, loss_conf, optimizer_conf, data_conf, batch_size, acc_grad_batches=1, hparams=None):
        if isinstance(model_conf, dict):
            arch = model_conf['arch'].split('-')
            if len(arch) == 1:
                arch = 'ap'
            else:
                model_conf['arch'] = arch[1]
                arch = arch[0]

            if arch == 'ap':
                for k in ('partial_residual',):
                    model_conf.pop(k)
                model = AstroPoint(**model_conf)
            elif arch == 'r2d2':
                for k in ('partial_residual',):
                    model_conf.pop(k)
                model_conf['des_head']['dimensions'] = 128
                model_conf['qlt_head']['single'] = loss_conf['loss_type'] != 'thresholded'
                model = R2D2(**model_conf)
            elif arch == 'mob':
                model = MobileAP(**model_conf)
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

    def update_param(self, param, value):
        p, ok = param.split('.'), True
        if p[0] == 'data':
            if p[1] in self.data_conf:
                self.data_conf[p[1]] = value
            else:
                ok = False
        else:
            ok = super(TerrestrialTrial, self).update_param(param, value)
        return ok

    def log_values(self):
        log = {}
        if not isinstance(self.loss_fn.wdt, float):
            log['wdt'] = torch.exp(-self.loss_fn.wdt)
        if not isinstance(self.loss_fn.wap, float):
            log['wap'] = torch.exp(-self.loss_fn.wap)
        if not isinstance(self.loss_fn.wqt, float):
            log['wqt'] = torch.exp(-self.loss_fn.wqt)
        if not isinstance(self.loss_fn.base, float):
            log['ap_base'] = self.loss_fn.base
        return log or None

    def resource_loss(self, loss):
        # TODO: use self.macs and self.target_macs
        return loss  # * some_good_fn(self.macs - self.target_macs)

    def build_training_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[0]

    def build_validation_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[1]

    def build_test_data_loader(self, rgb=False):
        return self._get_datasets(rgb)[2]

    def _get_datasets(self, rgb):
        if self._tr_data is None:
            npy = json.loads(self.data_conf['npy'])
            common = dict(margin=self.loss_fn.border, eval=False, rgb=rgb, npy=npy)
            dconf = {k: v for k, v in self.data_conf.items() if k in ('max_sc', 'noise_max', 'rnd_gain', 'image_size')}
            sconf = {k: v for k, v in self.data_conf.items() if k in ('max_rot', 'max_shear', 'max_proj')}
            sconf.update({'max_tr': 0, 'max_rot': math.radians(sconf['max_rot'])})

            ds = []
            if 1:
                ds.append(AachenFlowPairDataset(self.data_conf['path'], **common, **dconf))
            if 1:
                ds.append(WebImageSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                ds.append(AachenStyleTransferPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                ds.append(AachenSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))

            fullset = AugmentedConcatDataset(ds)
            datasets = fullset.split(self.data_conf.get('trn_ratio', 0.8),
                                     self.data_conf.get('val_ratio', 0.1),
                                     self.data_conf.get('tst_ratio', 0.1), eval=(1, 2))

            self._tr_data, self._val_data, self._test_data = \
                self.wrap_ds(datasets[0]), self.wrap_ds(datasets[1]), self.wrap_ds(datasets[2])
        return self._tr_data, self._val_data, self._test_data

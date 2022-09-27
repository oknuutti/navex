import json
import math
import os
from typing import Tuple

import torch
from torch import Tensor

from ..datasets.terrestrial.aachen import AachenFlowPairDataset, AachenSynthPairDataset, AachenStyleTransferPairDataset
from ..datasets.base import AugmentedConcatDataset, ShuffledDataset, split_tiered_data
from ..datasets.terrestrial.revisitop1m import WebImageSynthPairDataset
from ..losses.r2d2 import R2D2Loss
from ..models.astropoint import AstroPoint
from ..models.disk import DISK
from ..models.hynet import HyNet
from ..models.mobile_ap import MobileAP
from ..models.r2d2 import R2D2
from .base import TrialBase


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

            # False is always a bad idea, leads to polarized qlt output
            model_conf['qlt_head']['single'] = True
            for k in ('partial_residual',):     # unused, TODO: remove from definition.yaml
                model_conf.pop(k)

            if loss_conf['loss_type'] in ('disk', 'disk-p'):
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

        super(TerrestrialTrial, self).__init__(
            model=model,
            loss_fn=R2D2Loss(**loss_conf) if isinstance(loss_conf, dict) else loss_conf,
            optimizer_conf=optimizer_conf,
            acc_grad_batches=acc_grad_batches)

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
        for p, f in (('wdt', 'e'), ('wap', 'e'), ('wqt', 'e'), ('base', 'n'), ('ap_base', 'n')):
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

            dsp, dss = [], []
            if 1:
                dsp.append(AachenFlowPairDataset(self.data_conf['path'], **common, **dconf))
            if 1:
                dss.append(WebImageSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                dss.append(AachenStyleTransferPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                dss.append(AachenSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))

            trn, val, tst = split_tiered_data(dsp, dss, self.data_conf['trn_ratio'],
                                              self.data_conf['val_ratio'], self.data_conf['tst_ratio'])

            self._tr_data = self.wrap_ds(trn)
            self._val_data = self.wrap_ds(val)
            self._test_data = self.wrap_ds(tst)

        return self._tr_data, self._val_data, self._test_data

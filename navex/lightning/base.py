import threading
from typing import Union, Dict, Any, Optional
from argparse import Namespace

import torch
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import AsynchronousLoader

from ..trials.base import TrialBase


class TrialWrapperBase(pl.LightningModule):
    def __init__(self, trial: TrialBase, extra_hparams=None, use_gpu=True):
        super(TrialWrapperBase, self).__init__()
        self.trial = trial
        self.trial.loss_backward_fn = lambda x: None
        self.trial.step_optimizer_fn = lambda x: None
        self.trial.hparams.update(extra_hparams or {})
        self.save_hyperparameters(self.trial.hparams)
        self.use_gpu = use_gpu

    def forward(self, x):
        return self.trial.model(x)

    def configure_optimizers(self):
        return self.trial.optimizer

    def build_training_data_loader(self, rgb=False):
        return self._wrap_dl(self.trial.build_training_data_loader(rgb=rgb))

    def build_validation_data_loader(self, rgb=False):
        return self._wrap_dl(self.trial.build_validation_data_loader(rgb=rgb))

    def build_test_data_loader(self, rgb=False):
        return self._wrap_dl(self.trial.build_test_data_loader(rgb=rgb))

    def _wrap_dl(self, dl):
        return AsynchronousLoader(dl) if self.use_gpu else dl

    def training_step(self, batch, batch_idx):
        data, labels = batch
        epoch_id = self.trainer.current_epoch
        loss, output = self.trial.train_batch(data, labels, epoch_id, batch_idx)

        with torch.no_grad():
            acc = self.trial.accuracy(*output, labels, top_k=300, mutual=True, ratio=False, success_px_limit=12)
            self._log('trn', loss, acc, self.trial.log_values())

        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, 'tst')

    def _eval_step(self, batch, batch_idx, log_prefix):
        data, labels = batch
        loss, acc, output = self.trial.evaluate_batch(data, labels, top_k=300, mutual=True,
                                                      ratio=False, success_px_limit=12)
        self._log(log_prefix, loss, acc, self.trial.log_values())
        return loss

    def _log(self, lp, loss, acc, trial_params=None):
        tot, inl, dst, map = self.nanmean(acc)
        postfix = '_epoch' if lp == 'val' else ''

        log_values = {
            lp + '_loss' + postfix: loss,
            lp + '_tot' + postfix: tot * 100,
            lp + '_inl' + postfix: inl * 100,
            lp + '_dst' + postfix: dst,
            lp + '_map' + postfix: map * 100,
        }

        if trial_params is not None:
            log_values.update({'%s_%s%s' % (lp, p, postfix): v for p, v in trial_params.items()})

        # logger only
        self.log_dict(log_values, prog_bar=False, logger=True, on_step=None, on_epoch=True, reduce_fx=self.nanmean)

        # progress bar only
        self.log_dict({
            'tot': tot * 100,
            'inl': inl * 100,
            'dst': dst,
            'map': map * 100,
        }, prog_bar=True, logger=False, on_step=True, on_epoch=False, reduce_fx=self.nanmean)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def validation_epoch_end(self, outputs):
        val_losses = torch.stack(outputs)
        hp_metric = -self.nanmean(val_losses)
        self.log('hp_metric', hp_metric)

    @staticmethod
    def nanmean(x: Tensor):
        return torch.nansum(x, dim=0) / torch.sum(torch.logical_not(torch.isnan(x)), dim=0)


class MyLogger(TensorBoardLogger):
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        metrics = float('nan') if metrics is None else metrics
        return super(MyLogger, self).log_hyperparams(params, metrics)


class MySLURMConnector(SLURMConnector):
    def register_slurm_signal_handlers(self):
        if threading.current_thread() == threading.main_thread():
            super(MySLURMConnector, self).register_slurm_signal_handlers()

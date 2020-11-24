from typing import Union, Dict, Any, Optional
from argparse import Namespace

import torch
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ..trials.base import TrialBase


class TrialWrapperBase(pl.LightningModule):
    def __init__(self, trial: TrialBase, extra_hparams=None):
        super(TrialWrapperBase, self).__init__()
        self.trial = trial
        self.trial.loss_backward_fn = lambda x: None
        self.trial.step_optimizer_fn = lambda x: None
        self.trial.hparams.update(extra_hparams or {})
        self.save_hyperparameters(self.trial.hparams)

    def forward(self, x):
        return self.trial.model(x)

    def configure_optimizers(self):
        return self.trial.optimizer

    def training_step(self, batch, batch_idx):
        data, labels = batch
        epoch_id = self.trainer.current_epoch
        loss, output = self.trial.train_batch(data, labels, epoch_id, batch_idx)

        with torch.no_grad():
            acc = self.trial.accuracy(*output, labels, top_k=300, mutual=True, ratio=False, success_px_limit=12)
            self._log('trn', loss, acc)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, 'tst')

    def _eval_step(self, batch, batch_idx, log_prefix):
        data, labels = batch
        loss, acc, output = self.trial.evaluate_batch(data, labels, top_k=300, mutual=True,
                                                      ratio=False, success_px_limit=12)
        self._log(log_prefix, loss, acc)
        return loss

    def _log(self, lp, loss, acc):
        tot, inl, dst, map = self.nanmean(acc)
        postfix = '_epoch' if lp == 'val' else ''

        # logger only
        self.log_dict({
            lp + '_loss' + postfix: loss,
            lp + '_tot' + postfix: tot * 100,
            lp + '_inl' + postfix: inl * 100,
            lp + '_dst' + postfix: dst,
            lp + '_map' + postfix: map * 100,
        }, prog_bar=False, logger=True, on_step=None, on_epoch=True, reduce_fx=self.nanmean)

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


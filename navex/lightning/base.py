import threading
from typing import Union, Dict, Any, Optional
from argparse import Namespace

import torch
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import AsynchronousLoader

from ..trials.base import TrialBase, StudentTrialMixin


def _fn_none(x):
    return None


class TrialWrapperBase(pl.LightningModule):
    def __init__(self, trial: TrialBase = None, extra_hparams=None, use_gpu=True):
        super(TrialWrapperBase, self).__init__()
        self.use_gpu = use_gpu
        self.trial = trial
        if self.trial is not None:
            self.trial.loss_backward_fn = _fn_none
            self.trial.step_optimizer_fn = _fn_none
            self.trial.hparams.update(extra_hparams or {})
            self.save_hyperparameters(self.trial.hparams)
        self._restored_global_step = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint['trial'] = self.trial
        checkpoint['global_step'] = int(self.global_step)

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'].pop('trial.model.total_ops', None)
        checkpoint['state_dict'].pop('trial.model.total_params', None)
        checkpoint['state_dict'].pop('trial.model.backbone.total_ops', None)
        checkpoint['state_dict'].pop('trial.model.backbone.total_params', None)

        if self.trial is None and 'trial' in checkpoint:
            self.trial = checkpoint['trial']
        if 'global_step' in checkpoint:
            self._restored_global_step = int(checkpoint['global_step']) + 1

    def on_train_start(self):
        if self._restored_global_step is not None:
            self.trainer.global_step = self._restored_global_step
            self._restored_global_step = None

    def forward(self, x):
        return self.trial.model(x)

    @property
    def learning_rate(self):
        return self.trial.hparams['optimizer']['learning_rate']

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.trial.update_param('optimizer.learning_rate', learning_rate)
        self.log('learning_rate', learning_rate)

    def configure_optimizers(self):
        return self.trial.optimizer

    def build_training_data_loader(self, rgb=False):
        return self._wrap_dl(self.trial.build_training_data_loader(rgb=rgb))

    def build_validation_data_loader(self, rgb=False):
        return self._wrap_dl(self.trial.build_validation_data_loader(rgb=rgb))

    def build_test_data_loader(self, rgb=False):
        return self._wrap_dl(self.trial.build_test_data_loader(rgb=rgb))

    def wrap_ds(self, ds, shuffle=False):
        return self._wrap_dl(self.trial.wrap_ds(ds, shuffle=shuffle))

    def _wrap_dl(self, dl):
        return AsynchronousLoader(dl) if self.use_gpu else dl

    def training_step(self, batch, batch_idx):
        epoch_id = self.trainer.current_epoch

        if isinstance(self.trial, StudentTrialMixin):
            loss, output = self.trial.train_batch(batch, epoch_id, batch_idx, component_loss=True)
            output, labels = (output[0],), output[1]
        else:
            data, labels = batch
            loss, output = self.trial.train_batch(data, labels, epoch_id, batch_idx, component_loss=True)

        with torch.no_grad():
            acc = self.trial.accuracy(*output, labels, mutual=True, ratio=False, success_px_limit=5)
            self._log('trn', loss, acc, self.trial.log_values())

        return {'loss': loss.sum(dim=1), 'acc': acc}

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, 'tst')

    def _eval_step(self, batch, batch_idx, log_prefix):
        if isinstance(self.trial, StudentTrialMixin):
            loss, acc, output = self.trial.evaluate_batch(batch, mutual=True, ratio=False, success_px_limit=5,
                                                          component_loss=True)
        else:
            data, labels = batch
            loss, acc, output = self.trial.evaluate_batch(data, labels, mutual=True, ratio=False, success_px_limit=5,
                                                          component_loss=True)
        self._log(log_prefix, loss, acc, self.trial.log_values())
        return {'loss': loss.sum(dim=1), 'acc': acc}

    def _log(self, lp, loss, acc, trial_params=None):
        tot, inl, dst, map = self.nanmean(acc)
        postfix = '_epoch' if lp == 'val' else ''

        log_values = {
            lp + '_loss' + postfix: loss.sum(dim=1),
            lp + '_tot' + postfix: tot * 100,
            lp + '_inl' + postfix: inl * 100,
            lp + '_dst' + postfix: dst,
            lp + '_map' + postfix: map * 100,
        }

        if loss.shape[1] == 3:
            log_values[lp + '_des_loss' + postfix] = loss[:, 0]
            log_values[lp + '_det_loss' + postfix] = loss[:, 1]
            log_values[lp + '_qlt_loss' + postfix] = loss[:, 2]
        elif loss.shape[1] == 4:
            log_values[lp + '_peak_loss' + postfix] = loss[:, 0]
            log_values[lp + '_cosim_loss' + postfix] = loss[:, 1]
            log_values[lp + '_ap_loss' + postfix] = loss[:, 2]
            log_values[lp + '_qlt_loss' + postfix] = loss[:, 3]

        if hasattr(self.trial, 'resource_loss'):
            log_values[lp + '_rloss' + postfix] = self.trial.resource_loss(loss.sum(dim=1))

        if trial_params is not None:
            log_values.update({'%s_%s%s' % (lp, p, postfix): v for p, v in trial_params.items()})

        # logger only
        self.log_dict(log_values, prog_bar=False, logger=True, on_step=None, on_epoch=True, reduce_fx=self.nanmean)
        self.log_dict({'global_step': Tensor([self.trainer.global_step]).to(loss.device)},
                      prog_bar=False, logger=True, on_step=False, on_epoch=True, reduce_fx=torch.max)

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

    def training_epoch_end(self, outputs):
        loss = self.nanmean(torch.stack([o['loss'] for o in outputs]).flatten())
        tot, inl, dst, map = self.nanmean(torch.cat([o['acc'] for o in outputs], dim=0))
        self.trial.training_epoch_end(loss, tot, inl, dst, map)

    def validation_epoch_end(self, outputs):
        val_losses = torch.stack([o['loss'] for o in outputs])
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


class MyModelCheckpoint(ModelCheckpoint):
    def _save_top_k_checkpoints(self, trainer, pl_module, metrics):
        metrics = {k: v[0] if isinstance(v, Tensor) and len(v.shape) > 0 and len(v) == 1 else v for k, v in metrics.items()}
        super(MyModelCheckpoint, self)._save_top_k_checkpoints(trainer, pl_module, metrics)


class ValEveryNSteps(pl.Callback):
    def __init__(self, every_n_step):
        self.every_n_step = every_n_step

    def on_batch_end(self, trainer, pl_module):
        if trainer.global_step % self.every_n_step == 0 and trainer.global_step != 0:
            trainer.run_evaluation()
            trainer.logger_connector.set_stage("train")

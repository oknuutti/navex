import operator
import threading
from functools import reduce
from typing import Union, Dict, Any, Optional
from argparse import Namespace

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.training_loop import TrainLoop
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import AsynchronousLoader

from ..trials.base import TrialBase, StudentTrialMixin


def _fn_none(x):
    return None


class TrialWrapperBase(pl.LightningModule):
    def __init__(self, trial: TrialBase = None, extra_hparams=None, use_gpu=True,
                 hp_metric='val_loss_epoch', hp_metric_mode=-1):
        super(TrialWrapperBase, self).__init__()
        self.automatic_optimization = True
        self.use_gpu = use_gpu
        self.hp_metric = hp_metric
        self.hp_metric_mode = {'max': 1, 'min': -1}[hp_metric_mode] if isinstance(hp_metric_mode, str) else hp_metric_mode
        self.hp_metric_max = None
        self._restored_global_step = None

        self.trial = trial
        if self.trial is not None:
            if self.automatic_optimization:
                self.trial.loss_backward_fn = _fn_none
                self.trial.step_optimizer_fn = _fn_none
            self.trial.hparams.update(extra_hparams or {})
            self.save_hyperparameters(self.trial.hparams)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['trial'] = self.trial
        checkpoint['global_step'] = int(self.global_step)
        if self.hp_metric_max is not None:
            checkpoint['hp_metric_max'] = float(self.hp_metric_max)

    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'].pop('trial.model.total_ops', None)
        checkpoint['state_dict'].pop('trial.model.total_params', None)
        checkpoint['state_dict'].pop('trial.model.backbone.total_ops', None)
        checkpoint['state_dict'].pop('trial.model.backbone.total_params', None)

        if self.trial is None and 'trial' in checkpoint:
            self.trial = checkpoint['trial']
        if 'global_step' in checkpoint:
            self._restored_global_step = int(checkpoint['global_step']) + 1
        if 'hp_metric_max' in checkpoint:
            self.hp_metric_max = float(checkpoint['hp_metric_max'])

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
            loss, acc = self.trial.train_batch(batch, epoch_id, batch_idx, component_loss=True)
        else:
            data, labels = batch
            loss, acc = self.trial.train_batch(data, labels, epoch_id, batch_idx, component_loss=True)

        self._log('trn', loss, acc, self.trial.log_values())
        return {'loss': loss.sum(dim=1), 'acc': acc, 'losses': loss.detach()}

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if getattr(self.trial, 'on_train_batch_end'):
            acc = torch.cat([o['extra']['acc'] for oo in outputs for o in oo])
            losses = torch.cat([o['extra']['losses'] for oo in outputs for o in oo])
            self.trial.on_train_batch_end(losses, acc, self.trainer.train_loop.should_accumulate())

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, 'tst')

    def _eval_step(self, batch, batch_idx, log_prefix):
        if isinstance(self.trial, StudentTrialMixin):
            loss, acc, output = self.trial.evaluate_batch(batch, component_loss=True)
        else:
            data, labels = batch
            loss, acc, output = self.trial.evaluate_batch(data, labels, component_loss=True)
        self._log(log_prefix, loss, acc, self.trial.log_values())
        return {'loss': loss.sum(dim=1), 'acc': acc, 'losses': loss.detach()}

    def _gather_metrics(self, losses, acc, lp=None, extra=None):
        postfix = '_epoch' if lp == 'val' else ''
        lp = (lp + '_') if lp else ''
        tot, inl, dst, map = self.nanmean(acc)
        losses = torch.atleast_2d(self.nanmean(losses, dim=0))
        loss = losses.sum(dim=1)

        log_values = {
            lp + 'loss' + postfix: loss,
            lp + 'tot' + postfix: tot * 100,
            lp + 'inl' + postfix: inl * 100,
            lp + 'dst' + postfix: dst,
            lp + 'map' + postfix: map * 100,
        }

        if losses.shape[1] == 3:
            log_values.update({
                lp + 'des_loss' + postfix: losses[:, 0],
                lp + 'det_loss' + postfix: losses[:, 1],
                lp + 'qlt_loss' + postfix: losses[:, 2],
            })
        elif losses.shape[1] == 4:
            log_values.update({
                lp + 'peak_loss' + postfix: losses[:, 0],
                lp + 'cosim_loss' + postfix: losses[:, 1],
                lp + 'ap_loss' + postfix: losses[:, 2],
                lp + 'qlt_loss' + postfix: losses[:, 3],
            })

        if hasattr(self.trial, 'resource_loss'):
            log_values[lp + 'rloss' + postfix] = self.trial.resource_loss(losses.sum(dim=1))

        if extra is not None:
            log_values.update({lp + p + postfix: v for p, v in extra.items()})

        return log_values, (loss, tot, inl, dst, map)

    def _log(self, lp, losses, acc, trial_params=None):
        log_values, (loss, tot, inl, dst, map) = self._gather_metrics(losses, acc, lp=lp, extra=trial_params)

        # logger only, validation epoch end logging handled at self.validation_epoch_end(...)
        self.log_dict(log_values, on_step=None, on_epoch=(lp != 'val'), reduce_fx=self.nanmean)

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

    def _calc_hp_metric(self, metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        hpmval = 1
        for key in [m.strip() for m in self.hp_metric.split('*')]:
            try:
                hpmval *= float(key)
            except:
                if key in metrics:
                    hpmval *= metrics[key]
                else:
                    return None
        return (hpmval * self.hp_metric_mode) if isinstance(hpmval, torch.Tensor) else None

    def validation_epoch_end(self, outputs):
        metrics, _ = self._gather_metrics(torch.cat([o['losses'] for o in outputs], dim=0),
                                          torch.cat([o['acc'] for o in outputs], dim=0), lp='val')
        device = _[0].device
        metrics['global_step'] = Tensor([self.trainer.global_step + 1]).to(device)

        hpmval = self._calc_hp_metric(metrics)
        if hpmval is not None:
            self.hp_metric_max = hpmval.item() if self.hp_metric_max is None else max(self.hp_metric_max, hpmval.item())
            metrics.update({'hp_metric': hpmval, 'hp_metric_max': Tensor([self.hp_metric_max]).to(device)})

        self.log_dict(metrics)

    @staticmethod
    def nanmean(x: Tensor, dim=0):
        return torch.nansum(x, dim=dim) / torch.sum(torch.logical_not(torch.isnan(x)), dim=dim)


class MyLogger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        super(MyLogger, self).__init__(*args, **kwargs)
        self._hp_metric_initialized = False

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace],
                        metrics: Optional[Dict[str, Any]] = None) -> None:
        if metrics is None:
            if self._hp_metric_initialized:
                return
            metrics = {'hp_metric': float('nan')}
            self._hp_metric_initialized = True

        return super(MyLogger, self).log_hyperparams(params, metrics)


class MySLURMConnector(SLURMConnector):
    def register_slurm_signal_handlers(self):
        if threading.current_thread() == threading.main_thread():
            super(MySLURMConnector, self).register_slurm_signal_handlers()


class MyModelCheckpoint(ModelCheckpoint):
    def _save_top_k_checkpoints(self, trainer, pl_module, metrics):
        metrics = {k: v[0] if isinstance(v, Tensor) and len(v.shape) > 0 and len(v) == 1 else v for k, v in metrics.items()}
        super(MyModelCheckpoint, self)._save_top_k_checkpoints(trainer, pl_module, metrics)


class MyTrainLoop(TrainLoop):
    def should_check_val_fx(self, batch_idx, is_last_batch, on_epoch=True):
        # decide if we should run validation
        return self.trainer.enable_validation and not self.should_accumulate() \
               and (self.trainer.global_step + 1) % self.trainer.val_check_batch == 0


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MyTrainer, self).__init__(*args, **kwargs)
        self.train_loop = MyTrainLoop(self)

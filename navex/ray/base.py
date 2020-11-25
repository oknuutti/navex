import os
import re
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.callbacks import EarlyStopping

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from ..experiments.parser import set_nested, nested_update
from ..lightning.base import TrialWrapperBase, MyLogger
from ..trials.terrestrial import TerrestrialTrial


def execute_trial(hparams, checkpoint_dir=None, full_conf=None):
    # override full configuration set with contents of hparams
    # for key, val in hparams.items():
    #     set_nested(full_conf, key, val)
    nested_update(full_conf, hparams)

    # set paths
    sj_id = os.getenv('SLURM_JOB_ID')
    if sj_id is None:
        raise Exception('not a slurm node!')

    full_conf['data']['path'] = os.path.join('/tmp', sj_id, full_conf['data']['path'])
    full_conf['training']['output'] = tune.get_trial_dir()
    full_conf['training']['cache'] = os.path.join(full_conf['training']['output'], '..', 'cache')
    full_conf['model']['cache_dir'] = full_conf['training']['cache']

    train_conf = full_conf['training']

    version = None
    if checkpoint_dir:
        m = re.findall(r'(/|\\)version_(\d+)(/|\\|$)', checkpoint_dir)
        version = int(m[-1][1]) if m else None
    logger = MyLogger(train_conf['output'],
                      name=train_conf['name'], version=version)

    callbacks = [TuneReportCheckpointCallback(metrics={
        "loss": "val_loss_epoch",
        "tot_ratio": "val_tot_epoch",
        "inl_ratio": "val_inl_epoch",
        "px_err": "val_dst_epoch",
        "mAP": "val_map_epoch",
    }, filename="checkpoint", on="validation_end")]

    if train_conf['early_stopping']:
        callbacks.append(EarlyStopping(monitor='val_loss_epoch',
                                       mode='min',
                                       min_delta=0.00,
                                       patience=train_conf['early_stopping'],
                                       verbose=True))

    trainer = pl.Trainer(
        default_root_dir=tune.get_trial_dir(),  # was train_conf['output'],
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=train_conf['acc_grad_batches'],
        max_epochs=train_conf['epochs'],
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=train_conf['test_freq'],
        resume_from_checkpoint=train_conf.get('resume', None),
        log_every_n_steps=train_conf['print_freq'],
        flush_logs_every_n_steps=10,
        gpus=int(train_conf['gpu']),
        auto_select_gpus=bool(train_conf['gpu']),
        deterministic=bool(train_conf['deterministic']),
        auto_lr_find=bool(train_conf['auto_lr_find']),
        precision=16 if train_conf['gpu'] and train_conf['reduced_precision'] else 32
    )

    if checkpoint_dir:
        if 1:
            # Currently, this leads to errors (already fixed?):
            model = TrialWrapperBase.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))
        else:
            # Workaround:
            ckpt = pl_load(os.path.join(checkpoint_dir, "checkpoint"), map_location=lambda storage, loc: storage)
            model = TrialWrapperBase._load_model_state(ckpt, config=hparams)
            trainer.current_epoch = ckpt["epoch"]
    else:
        trial = TerrestrialTrial(full_conf['model'], full_conf['loss'], full_conf['optimizer'], full_conf['data'],
                                 train_conf['batch_size'], train_conf['acc_grad_batches'],
                                 hparams)
        model = TrialWrapperBase(trial)
        trn_dl = trial.build_training_data_loader()
        val_dl = trial.build_validation_data_loader()

    trainer.fit(model, trn_dl, val_dl)


def tune_asha(search_conf, hparams, full_conf):
    train_conf = full_conf['training']

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=train_conf['epochs'],
        grace_period=search_conf['grace_period'],
        reduction_factor=search_conf['reduction_factor'])

    reporter = CLIReporter(
        parameter_columns=list(hparams.keys())[:4],
        metric_columns=["loss", "inl_ratio", "mAP"])

    tune.run(
        partial(execute_trial, full_conf=full_conf),
        resources_per_trial={
            "cpu": full_conf['data']['workers'],
            "gpu": train_conf['gpu'],
        },
        config=hparams,
        # upload_dir=train_conf['output'],
        # trial_name_creator=,
        # trial_dirname_creator=,
        num_samples=search_conf['samples'],
        scheduler=scheduler,
        queue_trials=True,
        progress_reporter=reporter,
        name=train_conf['name'])

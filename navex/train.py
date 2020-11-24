
import os
import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .experiments.parser import ExperimentConfigParser, to_dict
from .trials.terrestrial import TerrestrialTrial
from .lightning.base import TrialWrapperBase, MyLogger


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    args = config.training

    os.makedirs(args.output, exist_ok=True)

    trial = TerrestrialTrial(to_dict(config.model), to_dict(config.loss),
                             to_dict(config.optimizer), to_dict(config.data),
                             args.batch_size, args.acc_grad_batches, to_dict(config.hparams))
    model = TrialWrapperBase(trial)
    trn_dl = trial.build_training_data_loader()
    val_dl = trial.build_validation_data_loader()

    version = None
    if args.resume:
        m = re.findall(r'-r(\d+)-', args.resume)
        version = int(m[-1]) if m else None
    logger = MyLogger(args.output, name=args.name, version=version)

    callbacks = [ModelCheckpoint(monitor='val_loss_epoch',
                                 mode='min',
                                 verbose=True,
                                 period=args.save_freq,
                                 # dirpath=args.output,  # by default: default_root_dir/name/version
                                 filename='%s-%s-r%d-{epoch}-{val_loss_epoch:.3f}'
                                          % (config.model.arch, args.name, logger.version))]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss_epoch',
                                       mode='min',
                                       min_delta=0.00,
                                       patience=args.early_stopping,
                                       verbose=True))

    trainer = pl.Trainer(default_root_dir=args.output,
                         logger=logger,
                         callbacks=callbacks,
                         accumulate_grad_batches=args.acc_grad_batches,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=args.print_freq,
                         check_val_every_n_epoch=args.test_freq,
                         resume_from_checkpoint=getattr(args, 'resume', None),
                         log_every_n_steps=args.print_freq,
                         flush_logs_every_n_steps=10,
                         gpus=1 if args.gpu else 0,
                         auto_select_gpus=bool(args.gpu),
                         deterministic=bool(args.deterministic),
                         auto_lr_find=bool(args.auto_lr_find),
                         precision=16 if args.gpu and args.reduced_precision else 32)
    trainer.fit(model, trn_dl, val_dl)

    tst_dl = trial.build_test_data_loader()
    trainer.test(model, test_dataloaders=tst_dl)


if __name__ == '__main__':
    main()

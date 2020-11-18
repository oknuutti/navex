
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from .experiments.parser import ExperimentConfigParser, to_dict
from .trials.terrestrial import TerrestrialTrial
from .lightning_api.base import TrialWrapperBase


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

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          period=args.save_freq,
                                          dirpath=args.output,
                                          filename='%s-%s-{epoch}-{val_loss:.3f}' % (config.model.arch, args.name))

    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=False,
                                        mode='max')

    use_gpu = 0

    trainer = pl.Trainer(default_root_dir=args.output,
                         logger=TensorBoardLogger(args.output, name=args.name),
                         callbacks=[checkpoint_callback, early_stop_callback],
                         accumulate_grad_batches=args.acc_grad_batches,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=args.print_freq,
                         check_val_every_n_epoch=args.test_freq,
                         resume_from_checkpoint=getattr(args, 'resume', None),
                         log_every_n_steps=2,
                         flush_logs_every_n_steps=10,
                         gpus=1 if use_gpu else 0,
                         auto_select_gpus=True if use_gpu else False,
                         deterministic=False,
                         auto_lr_find=False,
                         precision=16 if use_gpu else 32)
    trainer.fit(model, trn_dl, val_dl)

    tst_dl = trial.build_test_data_loader()
    trainer.test(model, test_dataloaders=tst_dl)


if __name__ == '__main__':
    main()

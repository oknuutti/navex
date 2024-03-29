
import os
import pickle
import re
import math
import sys
from shutil import copyfile

import torch

from pytorch_lightning.callbacks import EarlyStopping

from .trials.aerial import AerialTrial
from .trials.asteroidal import AsteroidalTrial
from .trials.asterostudent import AsteroStudentTrial
from .trials.terrestrial import TerrestrialTrial
from .trials.terrastudent import TerraStudentTrial
from .experiments.parser import ExperimentConfigParser, to_dict
from .lightning.base import TrialWrapperBase, MyLogger, MyModelCheckpoint, MyTrainer

PROFILING_ONLY = 0


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    parser = ExperimentConfigParser(definition=def_file)
    parser.add_argument('--preproc-path', default='', help="preprocess data into this folder, exit")

    config = parser.parse_args()
    args = config.training
    args.output = os.path.join(args.output, args.name)

    if PROFILING_ONLY:
       config.data.workers = 0

    os.makedirs(args.output, exist_ok=True)

    if args.gpu:
        totmem = torch.cuda.get_device_properties(0).total_memory  # in bytes
    else:
        import psutil
        totmem = psutil.virtual_memory().available  # in bytes
    totmem -= 256 * 1024 * 1024  # overhead
    acc_grad_batches = 2 ** max(0, math.ceil(math.log2((args.batch_mem * 1024 * 1024) / totmem)))  # in MB
    gpu_batch_size = args.batch_size // acc_grad_batches

    TrialClass = {cls.NAME: cls for cls in (
        TerrestrialTrial, TerraStudentTrial, AsteroidalTrial, AerialTrial, AsteroStudentTrial   #, AeroStudentTrial
    )}.get(args.trial, None)
    assert TrialClass is not None, 'invalid trial: %s' % args.trial

    trial = TrialClass(to_dict(config.model), to_dict(config.loss),
                       to_dict(config.optimizer), to_dict(config.data),
                       gpu_batch_size, acc_grad_batches, to_dict(config.hparams), to_dict(config.training.accuracy))

    model = TrialWrapperBase(trial, use_gpu=bool(args.gpu), hp_metric=config.search.metric,
                             hp_metric_mode=config.search.mode)
    if config.preproc_path:
        preprocess_data(model, config)
        return

    trn_dl = model.build_training_data_loader(rgb=config.model.in_channels == 3)
    val_dl = model.build_validation_data_loader(rgb=config.model.in_channels == 3)

    version = None
    if args.resume:
        m = re.findall(r'-r(\d+)-', args.resume)
        version = int(m[-1]) if m else None
    logger = MyLogger(args.output, name='', version=version)

    callbacks = [MyModelCheckpoint(monitor='hp_metric',
                                   mode='max',
                                   save_last=True,
                                   verbose=True,
                                   period=args.save_freq,
                                   dirpath=os.path.join(args.output, 'version_%d' % logger.version),
                                   filename='%s-%s-r%d-{global_step}-{hp_metric:.3f}'
                                             % (config.model.arch, args.name, logger.version)),
                ]

    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor='hp_metric',
                                       mode='max',
                                       min_delta=0.00,
                                       patience=args.early_stopping,
                                       verbose=True))

    trainer = MyTrainer(default_root_dir=args.output,
                        logger=logger,
                        callbacks=callbacks,
                        accumulate_grad_batches=acc_grad_batches,
                        max_steps=31 if PROFILING_ONLY else args.epochs,  # TODO (1): rename param
                        progress_bar_refresh_rate=args.print_freq,
                        check_val_every_n_epoch=sys.maxsize,
                        val_check_interval=min(args.test_freq, len(trn_dl)),
                        limit_train_batches=0.002 if PROFILING_ONLY else 1.0,
                        limit_val_batches=0.005 if args.test_freq <= 20 or PROFILING_ONLY else 1.0,
                        resume_from_checkpoint=getattr(args, 'resume', None),
                        log_every_n_steps=1,
                        flush_logs_every_n_steps=10,
                        gpus=1 if args.gpu else 0,
                        auto_select_gpus=bool(args.gpu),
                        deterministic=bool(args.deterministic),
                        auto_lr_find=bool(args.auto_lr_find),
                        precision=16 if args.gpu and args.reduced_precision else 32)

    if args.auto_lr_find == 1:
        trainer.tune(model, trn_dl, val_dl)
    elif args.auto_lr_find > 1:
        lr_finder = trainer.tuner.lr_find(model, trn_dl, val_dl, min_lr=1e-5, max_lr=1e-2)
        print('auto lr finder results: %s' % (lr_finder.results,))
        print('\nauto lr finder suggestion: %s' % (lr_finder.suggestion(),))
        path = os.path.join(args.output, args.name, 'lr_finder.pickle')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump(lr_finder, fh)
        print('saved lr_finder object in file %s' % (path,))
        return

    print('Starting to train (trn: %d, val: %d):' % (len(trn_dl), len(val_dl)))
    trainer.fit(model, trn_dl, val_dl)

    if PROFILING_ONLY:
        return

    tst_dl = model.build_test_data_loader(rgb=config.model.in_channels == 3)
    if tst_dl is not None:
        trainer.test(model, test_dataloaders=tst_dl)


def preprocess_data(model, config):
    import tqdm
    import logging
    logging.basicConfig(level=logging.INFO)

    rgb = config.model.in_channels == 3
    trn_dl = model.build_training_data_loader(rgb=rgb)
    val_dl = model.build_validation_data_loader(rgb=rgb)
    tst_dl = model.build_test_data_loader(rgb=rgb)

    def walk(p, dss):
        if hasattr(p, 'dataset'):
            walk(p.dataset, dss)
        elif hasattr(p, 'datasets'):
            for c in p.datasets:
                walk(c, dss)
        else:
            exists = False
            for ds in dss:
                exists = exists or p is ds
            if not exists:
                dss.append(p)

    dss = []
    for name, dl in {'trn': trn_dl, 'val': val_dl, 'tst': tst_dl}.items():
        walk(dl.dataset, dss)

    for ds in dss:
        if hasattr(ds, 'preproc_path'):
            logging.info('started with %s' % ds)

            outpath = os.path.join(config.preproc_path, ds.folder)
            os.makedirs(outpath, exist_ok=True)

            ds.preproc_path = config.preproc_path
            for i in tqdm.trange(len(ds)):
                imgs, aflow, *_ = ds[i]

            copyfile(os.path.join(ds.root, "dataset_all.sqlite"),
                     os.path.join(outpath, "dataset_all.sqlite"))

            with open(os.path.join(outpath, "preprocessed.flag"), 'w') as fh:
                fh.write('')


if __name__ == '__main__':
    main()

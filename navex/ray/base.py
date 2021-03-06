import copyreg
import json
import math
import socket
import os
import pickle
import logging
import sys
from functools import partial
from typing import Dict

import torch

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.callbacks import EarlyStopping

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.sample import Domain, Quantized, Float, Integer, Categorical, Normal, LogUniform
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback, TuneCallback
from ray.tune.suggest import BasicVariantGenerator, ConcurrencyLimiter
from ray.tune.suggest.skopt import SkOptSearch, logger as sk_logger
from ray.tune.suggest.variant_generator import parse_spec_vars
from ray.tune.utils import flatten_dict
from skopt import space as sko_sp

from ..experiments.parser import nested_update, split_double_samplers
from ..lightning.base import TrialWrapperBase, MySLURMConnector, MyModelCheckpoint, MyTrainer, MyLogger
from ..models.tools import ordered_nested_dict, reorder_cols
from ..trials.aerial import AerialTrial
from ..trials.asteroidal import AsteroidalTrial
from ..trials.terrastudent import TerraStudentTrial
from ..trials.terrestrial import TerrestrialTrial

DEBUG = 0


def execute_trial(hparams, checkpoint_dir=None, full_conf=None, update_conf=False,
                  hp_metric='val_tot_epoch', hp_metric_mode='max'):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    job_id = os.getenv('SLURM_JOB_ID', '--')
    data_dir = os.getenv('NAVEX_DATA', '--')   # '/tmp/navex'

    logging.info(f'at execute_trial function, {host} ({ip}), job_id: {job_id}, data: {data_dir}')

    # override full configuration set with contents of hparams
    nested_update(full_conf, hparams)

    # set paths
    full_conf['data']['path'] = data_dir
    full_conf['training']['output'] = tune.get_trial_dir()
    full_conf['training']['cache'] = os.path.join(full_conf['training']['output'], '..', 'cache')
    full_conf['model']['cache_dir'] = full_conf['training']['cache']

    train_conf = full_conf['training']

    # version is always '' as output dir is unique for each trial
    logger = MyLogger(train_conf['output'], name='', version='')

    callbacks = [
        # on_validation_end, run #1
        MyModelCheckpoint(monitor='hp_metric',
                          mode='max',
                          verbose=True,
                          dirpath=os.path.join(train_conf['output'], 'version_%s' % logger.version),
                          filename='checkpoint-{global_step}-{hp_metric:.3f}'),

        # on_validation_end, run #2
        TuneReportCheckpointCallback(metrics={
            # "rloss": "val_rloss_epoch",
            "loss": "val_loss_epoch",
            "tot_ratio": "val_tot_epoch",
            "inl_ratio": "val_inl_epoch",
            "px_err": "val_dst_epoch",
            "mAP": "val_map_epoch",
            "global_step": "global_step",
            "hp_metric": "hp_metric",
            "hp_metric_max": "hp_metric_max",
        }, filename="checkpoint", on="validation_end"),
    ]

    if train_conf['early_stopping']:
        callbacks.append(EarlyStopping(monitor='hp_metric',
                                       mode='max',
                                       min_delta=0.00,
                                       patience=train_conf['early_stopping'],
                                       verbose=True))

    totmem = torch.cuda.get_device_properties(0).total_memory  # in bytes
    totmem -= 256 * 1024 * 1024  # overhead
    acc_grad_batches = 2 ** max(0, math.ceil(math.log2((train_conf['batch_mem'] * 1024 * 1024) / totmem)))  # in MB
    gpu_batch_size = train_conf['batch_size'] // acc_grad_batches

    trainer = MyTrainer(
        default_root_dir=tune.get_trial_dir(),  # was train_conf['output'],
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=acc_grad_batches,
        max_steps=train_conf['epochs'],  # TODO (1): rename param
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=sys.maxsize,
        val_check_interval=train_conf['test_freq'],
        limit_train_batches=0.002 if DEBUG else 1.0,
        limit_val_batches=0.004 if DEBUG else 1.0,
        resume_from_checkpoint=train_conf.get('resume', None),
        log_every_n_steps=1,
        flush_logs_every_n_steps=10,
        gpus=int(train_conf['gpu']),
        auto_select_gpus=bool(train_conf['gpu']),
        deterministic=bool(train_conf['deterministic']),
        auto_lr_find=bool(train_conf['auto_lr_find']),
        precision=16 if train_conf['gpu'] and train_conf['reduced_precision'] else 32
    )
    # Workaround for problem where signal.signal(signal.SIGUSR1, self.sig_handler) throws error
    #   "ValueError: signal only works in main thread", means that lightning based rescheduling won't work
    trainer.slurm_connector = MySLURMConnector(trainer)

    if checkpoint_dir:
        logging.info('restoring checkpoint from %s' % (checkpoint_dir,))
        if 0:
            # Currently, this leads to errors (already fixed?):
            model = TrialWrapperBase.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint"),
                                                          map_location="cuda:0" if int(train_conf['gpu']) else "cpu")
        else:
            # Workaround:
            ckpt = pl_load(os.path.join(checkpoint_dir, "checkpoint"),
                           map_location="cuda:0" if int(train_conf['gpu']) else "cpu")
            model = TrialWrapperBase._load_model_state(ckpt)
            model.trial._tr_data = None
            model.trial.update_conf(new_conf={'data.path': full_conf['data']['path']})
            trainer.current_epoch = ckpt["epoch"]

        if update_conf:
            model.trial.update_conf(new_conf=hparams, fail_silently=False)

    else:
        logging.info('npy is %s' % (json.dumps(json.loads(full_conf['data']['npy'])),))
        logging.info('new trial with %s' % (full_conf,))

        TrialClass = {cls.NAME: cls for cls in (
            TerrestrialTrial, TerraStudentTrial, AsteroidalTrial, AerialTrial,  # AstraStudentTrial, AeroStudentTrial
        )}.get(train_conf['trial'], None)
        assert TrialClass is not None, 'invalid trial: %s' % train_conf['trial']
        trial = TrialClass(full_conf['model'], full_conf['loss'], full_conf['optimizer'], full_conf['data'],
                           gpu_batch_size, acc_grad_batches, hparams)
        model = TrialWrapperBase(trial)

    model.hp_metric, model.hp_metric_mode = hp_metric, {'max': 1, 'min': -1}[hp_metric_mode]
    trn_dl = model.trial.build_training_data_loader()
    val_dl = model.trial.build_validation_data_loader()

    if not checkpoint_dir and train_conf['auto_lr_find']:
        # find optimal learning rate
        trainer.tune(model, trn_dl, val_dl)

    logging.info('start training with trn data len=%d and val data len=%d (path: %s)' % (len(trn_dl), len(val_dl),
                                                                                         model.trial.data_conf['path']))
    try:
        trainer.fit(model, trn_dl, val_dl)
    except Exception as e:
        logging.error('encountered error %s' % e)
        raise e


def tune_asha(search_conf, hparams, full_conf):
    train_conf = full_conf['training']

    tmp = search_conf['method'].split('-')
    search_method = 'rs' if len(tmp) == 1 else tmp[1]

    if search_method == 'rs':
        search_alg = BasicVariantGenerator()
    elif search_method == 'bo':
        # need to:  pip install scikit-optimize
        initial, hparams = split_double_samplers(hparams)
        hparams = ordered_nested_dict(hparams)

        initial = flatten_dict(initial, prevent_delimiter=True)
        key_order = list(flatten_dict(hparams, prevent_delimiter=True).keys())
        if search_conf['resume']:
            search_alg = MySkOptSearch()
            search_alg.restore(search_conf['resume'])
            start_config = reorder_cols(search_alg._skopt_opt.Xi, search_alg._parameters, key_order)
            evaluated_rewards = search_alg._skopt_opt.yi
        else:
            start_config = [[initial[k].sample() for k in key_order if k in initial]
                             for _ in range(max(1, search_conf['nodes']))]
            evaluated_rewards = None

        search_alg = MySkOptSearch(metric='hp_metric_max', mode='max',
                                   points_to_evaluate=start_config, evaluated_rewards=evaluated_rewards)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max(1, search_conf['nodes']))
    else:
        assert False, ('Invalid search method "%s", only random search (rs) '
                       'or bayesian optimization (bo) supported') % tmp[1]

    scheduler = ASHAScheduler(
        time_attr="global_step",    # the default is "training_iteration" which equals epoch!!
        metric='hp_metric',
        mode='max',
        max_t=train_conf['epochs'],         # TODO: (1) change name of this config variable
        grace_period=search_conf['grace_period'],
        reduction_factor=search_conf['reduction_factor'])

    class MyReporter(CLIReporter):
        def report(self, trials, done, *sys_info):
            logging.info(self._progress_str(trials, done, *sys_info))

    reporter = MyReporter(
        parameter_columns=list(hparams.keys())[:4],
        metric_columns=["loss", "tot_ratio", "mAP"])

    tune.run(
        partial(execute_trial, full_conf=full_conf, update_conf=False,
                hp_metric=search_conf['metric'], hp_metric_mode=search_conf['mode']),
        resources_per_trial={
            "cpu": full_conf['data']['workers'],
            "gpu": train_conf['gpu'],
        },
        config=hparams,
        search_alg=search_alg,
        # upload_dir=train_conf['output'],
        # trial_name_creator=,
        # trial_dirname_creator=,
        num_samples=search_conf['samples'],
        scheduler=scheduler,
        queue_trials=True,
        reuse_actors=False,     # not sure if setting this True results in trials that are forever pending, True helps with fd limits though
        max_failures=20,
#        resume=search_conf['resume'].upper() or False, # ray tune still has too many bugs with resume, at least v1.1.0 doesnt work
#        local_dir=search_conf['results_path'] or None, # defaults to ~/ray_results
        # checkpoint_freq=200,      # if use functional api, this wont have any effect
        # checkpoint_at_end=True,   # if use functional api, this wont have any effect
        keep_checkpoints_num=1,
#        checkpoint_score_attr=f"min-{metric}" if mode == 'min' else metric,     # doesn't seem to work if not default
        progress_reporter=reporter,
        name=train_conf['name'])

    if search_method == 'bo':
        logging.info('RESULT: %s' % (search_alg.searcher._skopt_opt.get_result(),))
        logging.info('Length scales: %s' % (
            dict(zip(key_order, search_alg.searcher._skopt_opt.base_estimator_.kernel.k2.length_scale)),))


def tune_pbs(search_conf, hparams, full_conf):
    train_conf = full_conf['training']
    hparams, mutations = split_double_samplers(hparams)

    scheduler = PopulationBasedTraining(
        time_attr="global_step",
        perturbation_interval=search_conf['grace_period'],
        hyperparam_mutations=mutations,
        metric='hp_metric',
        mode='max'
    )

    class MyReporter(CLIReporter):
        def report(self, trials, done, *sys_info):
            logging.info(self._progress_str(trials, done, *sys_info))

    reporter = MyReporter(
        parameter_columns=list(hparams.keys())[:4],
        metric_columns=["loss", "tot_ratio", "mAP"])

    tune.run(
        partial(execute_trial, full_conf=full_conf, update_conf=True,
                hp_metric=search_conf['metric'], hp_metric_mode=search_conf['mode']),
        resources_per_trial={
            "cpu": full_conf['data']['workers'],
            "gpu": train_conf['gpu'],
        },
        config=hparams,
        num_samples=search_conf['samples'],
        scheduler=scheduler,
        queue_trials=True,
        reuse_actors=False,
        max_failures=20,
        keep_checkpoints_num=1,
        progress_reporter=reporter,
        name=train_conf['name'])


def sample(config, sampled=None):
    sampled = sampled or {}
    for key, distribution in config.items():
        if isinstance(distribution, dict):
            sampled[key] = sample(distribution)
        else:
            sampled[key] = distribution.sample()
    return sampled


class MySkOptSearch(SkOptSearch):

    def setup_skopt(self):
        self._parameter_names = list(self._parameter_names)
        self._parameter_ranges = list(self._parameter_ranges)
        super(MySkOptSearch, self).setup_skopt()

    def save(self, checkpoint_path: str):
        trials_object = (self._parameters, self._initial_points, self._skopt_opt)
        with open(checkpoint_path, "wb") as outputFile:
            p = pickle.Pickler(outputFile)

            # because somewhere in SkOptSearch there's dict_keys and dict_values objects:
            p.dispatch_table = copyreg.dispatch_table.copy()
            p.dispatch_table[{}.keys().__class__] = lambda x: (list, (list(x),))
            p.dispatch_table[{}.values().__class__] = lambda x: (list, (list(x),))

            p.dump(trials_object)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            trials_object = pickle.load(inputFile)
        i = 0
        if len(trials_object) == 3:
            self._parameters = trials_object[0]
            i = 1
        self._initial_points = trials_object[0 + i]
        self._skopt_opt = trials_object[1 + i]

    @staticmethod
    def convert_search_space(spec: Dict, join: bool = False) -> Dict:
        spec = flatten_dict(spec, prevent_delimiter=False)  # dont prevent delimeter!
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        if grid_vars:
            raise ValueError(
                "Grid search parameters cannot be automatically converted "
                "to a SkOpt search space.")

        def resolve_value(name: str, domain: Domain) -> sko_sp.Dimension:
            sampler = domain.get_sampler()
            if isinstance(sampler, Quantized):
                sk_logger.warning("SkOpt search does not support quantization. Dropped quantization.")
                sampler = sampler.get_sampler()

            if isinstance(sampler, Normal):
                sk_logger.warning(
                    "SkOpt does not support sampling from normal distribution."
                    " The {} sampler will be dropped.".format(sampler))

            prior = 'log-uniform' if sampler is not None and isinstance(sampler, LogUniform) else 'uniform'

            if isinstance(domain, Float):
                return sko_sp.Real(domain.lower, domain.upper, prior=prior, name=name)
            if isinstance(domain, Integer):
                return sko_sp.Integer(domain.lower, domain.upper, prior=prior, name=name)
            if isinstance(domain, Categorical):
                return sko_sp.Categorical(domain.categories, name=name)

            raise ValueError("SkOpt does not support parameters of type "
                             "`{}`".format(type(domain).__name__))

        # Parameter name is e.g. "a/b/c" for nested dicts
        space = {
            "/".join(path): resolve_value("/".join(path), domain)
            for path, domain in domain_vars
        }

        if join:
            spec.update(space)
            space = spec

        return space

    def _process_result(self, trial_id: str, result: Dict):
        logging.info('trial %s result: %s' % (trial_id, result))
        super(MySkOptSearch, self)._process_result(trial_id, result)

        # class _MyTuneCheckpointCallback(_TuneCheckpointCallback):
#     def _handle(self, trainer: Trainer, pl_module: LightningModule):
#         if trainer.running_sanity_check:
#             return
#         with tune.checkpoint_dir(step=trainer.global_step) as checkpoint_dir:
#             node_id = ray.get_runtime_context().node_id
#             terminate = ray.get(ray.get_actor('term_' + node_id).is_set.remote())
#             if terminate:
#                 logging.warning('should save as node %s will terminate soon!' % node_id)
#                 # TODO: what should do here?
#
#             trainer.save_checkpoint(
#                 os.path.join(checkpoint_dir, self._filename))
#
# class MyTuneReportCheckpointCallback(TuneReportCheckpointCallback):
#     def __init__(self,
#                  metrics: Union[None, str, List[str], Dict[str, str]] = None,
#                  filename: str = "checkpoint",
#                  on: Union[str, List[str]] = "validation_end"):
#         super(MyTuneReportCheckpointCallback, self).__init__(metrics, filename, on)
#         self._checkpoint = _MyTuneCheckpointCallback(filename, on)


class CheckOnSLURM(TuneCallback):
    def __init__(self):
        super(CheckOnSLURM, self).__init__('batch_end')

    def _handle(self, trainer: Trainer, pl_module: LightningModule):
        node_id = ray.get_runtime_context().node_id
        terminate = ray.get(ray.get_actor('term_' + node_id).is_set.remote())
        if terminate:
            trainer.checkpoint_connector.hpc_save(trainer.weights_save_path, trainer.logger)


# @ray.remote(num_cpus=0)
# class Signal:
#     def __init__(self):
#         self._flag = False
#
#     def set(self):
#         self._flag = True
#
#     def clear(self):
#         self._flag = False
#
#     def is_set(self):
#         return self._flag
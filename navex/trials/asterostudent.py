import json
import math

import torch

from ..lightning.base import TrialWrapperBase
from ..trials.asteroidal import AsteroidalTrial
from ..losses.student import StudentLoss

from ..datasets.base import AugmentedConcatDataset
from ..datasets.asteroidal.eros import ErosDataset
from ..datasets.asteroidal.itokawa import ItokawaDataset
from ..datasets.asteroidal.cg67p import CG67pNavcamDataset, CG67pOsinacDataset
from ..datasets.asteroidal.bennu import BennuDataset
from ..datasets.asteroidal.synth import SynthBennuDataset

from .base import StudentTrialMixin


class AsteroStudentTrial(StudentTrialMixin, AsteroidalTrial):
    NAME = 'astst'

    def __init__(self, model_conf, loss_conf, optimizer_conf, data_conf, batch_size, acc_grad_batches=1,
                 hparams=None, accuracy_params=None):
        # load teacher
        teacher_ckpt = loss_conf.pop('teacher')
        model = TrialWrapperBase.load_from_checkpoint(teacher_ckpt)
        teacher = model.trial.model

        accuracy_params = accuracy_params or {}     # TODO: make configurable from the .yaml file
        accuracy_params.setdefault('det_lim', 0.1)
        accuracy_params.setdefault('qlt_lim', 0.1)
        loss_conf['skip_qlt'] = model_conf['qlt_head']['skip']
        AsteroidalTrial.__init__(self, model_conf,
                StudentLoss(**loss_conf) if isinstance(loss_conf, dict) else loss_conf,
                optimizer_conf, data_conf, batch_size, acc_grad_batches, hparams, accuracy_params)

        StudentTrialMixin.__init__(self, teacher=teacher)
        self.target_macs = 0.5e9     # TODO: set at e.g. loss_conf

    def log_values(self):
        log = {}
        if not isinstance(self.loss_fn.des_w, float):
            log['des_w'] = torch.exp(-self.loss_fn.des_w)
        if not isinstance(self.loss_fn.det_w, float):
            log['det_w'] = torch.exp(-self.loss_fn.det_w)
        if not isinstance(self.loss_fn.qlt_w, float):
            log['qlt_w'] = torch.exp(-self.loss_fn.qlt_w)
        return log or None

    def _get_datasets(self, rgb=False):
        if self._tr_data is None:
            dconf = {k: v for k, v in self.data_conf.items()
                          if k in ('noise_max', 'rnd_gain', 'image_size', 'max_rot', 'max_shear', 'max_proj',
                                   'student_rnd_gain', 'student_noise_sd')}
            dconf.update(dict(eval=False, rgb=rgb, npy=json.loads(self.data_conf['npy']),
                              max_tr=0, max_rot=math.radians(dconf['max_rot'])))

            datasets = [ErosDataset,
                        BennuDataset,
                        CG67pNavcamDataset,
                        CG67pOsinacDataset]
            if self.data_conf['use_synth']:
                datasets += [SynthBennuDataset]

            fullset = AugmentedConcatDataset([cls(self.data_conf['path'], **dconf) for cls in datasets])
            trn, val = fullset.split(self.data_conf['trn_ratio'], 1 - self.data_conf['trn_ratio'], eval=(1,))
            tst = ItokawaDataset(self.data_conf['path'], **dconf)

            self._tr_data, self._val_data, self._test_data = map(self.wrap_ds, (trn, val, tst))
        return self._tr_data, self._val_data, self._test_data

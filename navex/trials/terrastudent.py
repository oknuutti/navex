import json
import math

import torch

from .base import StudentTrialMixin
from ..datasets.base import AugmentedConcatDataset
from ..datasets.terrestrial.aachen import AachenDataset, AachenSyntheticNightDataset
from ..datasets.terrestrial.revisitop1m import WebImageDataset
from ..lightning.base import TrialWrapperBase
from ..models.r2d2orig import R2D2
from ..trials.terrestrial import TerrestrialTrial
from ..losses.student import StudentLoss


class TerraStudentTrial(StudentTrialMixin, TerrestrialTrial):
    NAME = 'terrst'

    def __init__(self, model_conf, loss_conf, optimizer_conf, data_conf, batch_size, acc_grad_batches=1, hparams=None):
        # load teacher
        teacher_ckpt = loss_conf.pop('teacher')
        if teacher_ckpt[-5:] == '.ckpt':
            model = TrialWrapperBase.load_from_checkpoint(teacher_ckpt)
            teacher = model.trial.model
        else:
            teacher = R2D2(path=teacher_ckpt)

        loss_conf['skip_qlt'] = model_conf['qlt_head']['skip']
        TerrestrialTrial.__init__(self, model_conf,
                StudentLoss(**loss_conf) if isinstance(loss_conf, dict) else loss_conf,
                optimizer_conf, data_conf, batch_size, acc_grad_batches, hparams)

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

    def _get_datasets(self, rgb):
        if self._tr_data is None:
            dconf = {k: v for k, v in self.data_conf.items()
                          if k in ('noise_max', 'rnd_gain', 'image_size', 'max_rot', 'max_shear', 'max_proj')}
            dconf.update(dict(eval=False, rgb=rgb, npy=json.loads(self.data_conf['npy']),
                              max_tr=0, max_rot=math.radians(dconf['max_rot'])))

            ds = []
            if 1:
                ds.append(AachenDataset(self.data_conf['path'], **dconf))
            if 1:
                ds.append(AachenSyntheticNightDataset(self.data_conf['path'], **dconf))
            if 1:
                ds.append(WebImageDataset(self.data_conf['path'], **dconf))

            fullset = AugmentedConcatDataset(ds)
            datasets = fullset.split(self.data_conf.get('trn_ratio', 0.8),
                                     self.data_conf.get('val_ratio', 0.1),
                                     self.data_conf.get('tst_ratio', 0.1), eval=(1, 2))

            self._tr_data, self._val_data, self._test_data = \
                self.wrap_ds(datasets[0]), self.wrap_ds(datasets[1]), self.wrap_ds(datasets[2])
        return self._tr_data, self._val_data, self._test_data

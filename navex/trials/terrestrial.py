import json
import math

from ..datasets.terrestrial.aachen import AachenFlowPairDataset, AachenSynthPairDataset, AachenStyleTransferPairDataset
from ..datasets.base import split_tiered_data
from ..datasets.terrestrial.revisitop1m import WebImageSynthPairDataset
from .base import TrialBase


class TerrestrialTrial(TrialBase):
    NAME = 'terr'

    def _get_datasets(self, rgb):
        if self._tr_data is None:
            npy = json.loads(self.data_conf['npy'])
            common = dict(margin=self.loss_fn.border, eval=False, rgb=rgb, npy=npy)
            dconf = {k: v for k, v in self.data_conf.items() if k in ('max_sc', 'noise_max', 'rnd_gain', 'image_size')}
            sconf = {k: v for k, v in self.data_conf.items() if k in ('max_rot', 'max_shear', 'max_proj')}
            sconf.update({'max_tr': 0, 'max_rot': math.radians(sconf['max_rot'])})

            dsp, dss = [], []
            if 1:
                dsp.append(AachenFlowPairDataset(self.data_conf['path'], **common, **dconf))
            if 1:
                dss.append(WebImageSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                dss.append(AachenStyleTransferPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                dss.append(AachenSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))

            trn, val, tst = split_tiered_data(dsp, dss, self.data_conf['trn_ratio'],
                                              self.data_conf['val_ratio'], self.data_conf['tst_ratio'])

            self._tr_data = self.wrap_ds(trn)
            self._val_data = self.wrap_ds(val)
            self._test_data = self.wrap_ds(tst)

        return self._tr_data, self._val_data, self._test_data

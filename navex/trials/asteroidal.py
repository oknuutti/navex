import math

from .terrestrial import TerrestrialTrial
from ..datasets.asteroidal.eros import ErosPairDataset
from ..datasets.asteroidal.synth import SynthBennuPairDataset
from ..datasets.asteroidal.bennu import BennuSynthPairDataset
from ..datasets.asteroidal.cg67p import CG67pNavcamSynthPairDataset, CG67pOsinacPairDataset
from ..datasets.asteroidal.itokawa import ItokawaPairDataset
from ..datasets.base import split_tiered_data


class AsteroidalTrial(TerrestrialTrial):
    NAME = 'ast'

    def _get_datasets(self, rgb=False):
        assert rgb is False, 'no rgb images from asteroids'

        if self._tr_data is None:
            common = dict(margin=self.loss_fn.border, eval=False, rgb=False)
            common.update({k: v for k, v in self.data_conf.items() if k in ('noise_max', 'rnd_gain', 'image_size')})
            pconf = dict(aflow_rot_norm=True)
            sconf = {k: v for k, v in self.data_conf.items() if k in ('max_sc', 'max_rot', 'max_shear', 'max_proj')}
            sconf.update({'max_tr': 0, 'max_rot': math.radians(sconf['max_rot'])})

            dsp, dss = [], []
            if 1:
                dsp.append(ErosPairDataset(self.data_conf['path'], **common, **pconf))
            if 1:
                dsp.append(SynthBennuPairDataset(self.data_conf['path'], **common))
            if 1:
                dss.append(BennuSynthPairDataset(self.data_conf['path'], **common, **sconf))
            if 1:
                dss.append(CG67pNavcamSynthPairDataset(self.data_conf['path'], **common, **sconf))
            if 1:
                dsp.append(CG67pOsinacPairDataset(self.data_conf['path'], **common, **pconf))

            tst = ItokawaPairDataset(self.data_conf['path'], **common, **pconf)

            if 0:
                dsp.append(tst)
                tst = None

            trn, val, _ = split_tiered_data(dsp, dss, self.data_conf['trn_ratio'], self.data_conf['val_ratio'], 0)
            self._tr_data = self.wrap_ds(trn)
            self._val_data = self.wrap_ds(val)
            self._test_data = self.wrap_ds(tst)

        return self._tr_data, self._val_data, self._test_data

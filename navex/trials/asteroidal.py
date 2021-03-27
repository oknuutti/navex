import math

from .terrestrial import TerrestrialTrial
from ..datasets.asteroidal.eros import ErosPairDataset
from ..datasets.asteroidal.synth import SynthBennuPairDataset
from ..datasets.asteroidal.bennu import BennuSynthPairDataset
from ..datasets.asteroidal.cg67p import CG67pNavcamSynthPairDataset, CG67pOsinacPairDataset
from ..datasets.asteroidal.itokawa import ItokawaPairDataset
from ..datasets.base import AugmentedConcatDataset


class AsteroidalTrial(TerrestrialTrial):
    def _get_datasets(self, rgb=False):
        assert rgb is False, 'no rgb images from asteroids'

        if self._tr_data is None:
            common = dict(margin=self.loss_fn.ap_loss.super.sampler.border, eval=False, rgb=False)
            dconf = {k: v for k, v in self.data_conf.items() if k in ('noise_max', 'rnd_gain', 'image_size')}
            sconf = {k: v for k, v in self.data_conf.items() if k in ('max_rot', 'max_shear', 'max_proj')}
            sconf.update({'max_tr': 0, 'max_rot': math.radians(sconf['max_rot'])})

            ds = []
            if 1:
                ds.append(ErosPairDataset(self.data_conf['path'], **common, **dconf))
            if 0:
                ds.append(SynthBennuPairDataset(self.data_conf['path'], **common, **dconf))
            if 1:
                ds.append(BennuSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                ds.append(CG67pNavcamSynthPairDataset(self.data_conf['path'], **common, **sconf, **dconf))
            if 1:
                ds.append(CG67pOsinacPairDataset(self.data_conf['path'], **common, **dconf))

            tst_set = ItokawaPairDataset(self.data_conf['path'], **common, **dconf)

            if 0:
                ds.append(tst_set)
                tst_set = None

            fullset = AugmentedConcatDataset(ds)
            trn_set, val_set = fullset.split(self.data_conf.get('trn_ratio', 0.85),
                                             self.data_conf.get('val_ratio', 0.15), eval=(1,))

            self._tr_data, self._val_data, self._test_data = map(self.wrap_ds, (trn_set, val_set, tst_set))
        return self._tr_data, self._val_data, self._test_data

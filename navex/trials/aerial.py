import math

from .terrestrial import TerrestrialTrial
from ..datasets.aerial.batvik import BatvikSynthPairDataset, BatvikPairDataset
from ..datasets.base import AugmentedConcatDataset


class AerialTrial(TerrestrialTrial):
    NAME = 'aer'

    def _get_datasets(self, rgb=False):
        if self._tr_data is None:
            common = dict(margin=self.loss_fn.border, eval=False, rgb=False)
            common.update({k: v for k, v in self.data_conf.items() if k in ('noise_max', 'rnd_gain', 'image_size')})
            sconf = {k: v for k, v in self.data_conf.items() if k in ('max_sc', 'max_rot', 'max_shear', 'max_proj')}
            sconf.update({'max_tr': 0, 'max_rot': math.radians(sconf['max_rot'])})

            ds = []
            if 0:
                ds.append(BatvikSynthPairDataset(self.data_conf['path'],
                                                 subset=('14', '18', '20', '24', '31', '33'), **common, **sconf))
            if 1:
                common['noise_max'] = 0.0
                common['rnd_gain'] = 1.0
                pconf = dict(fixed_ground_res=0.25,  # 0.25; None
                             resize_max_sc=1.0)
                ds.append(BatvikPairDataset(self.data_conf['path'], **common, **pconf))

            if 1:
                tst_set = None
            else:
                tst_set = BatvikSynthPairDataset(self.data_conf['path'], subset=('2020-12-17',), **common)
                if 0:
                    ds.append(tst_set)
                    tst_set = None

            fullset = AugmentedConcatDataset(ds)
            trn_set, val_set = fullset.split(self.data_conf.get('trn_ratio', 0.90),
                                             self.data_conf.get('val_ratio', 0.10), eval=(1,))

            self._tr_data, self._val_data, self._test_data = map(self.wrap_ds, (trn_set, val_set, tst_set))
        return self._tr_data, self._val_data, self._test_data

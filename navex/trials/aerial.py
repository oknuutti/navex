import math

from .terrestrial import TerrestrialTrial
from ..datasets.aerial.batvik import BatvikSynthPairDataset, BatvikPairDataset
from ..datasets.aerial.gearth import GoogleEarthPairDataset
from ..datasets.base import AugmentedConcatDataset, RedundantlySampledDataset


class AerialTrial(TerrestrialTrial):
    NAME = 'aer'

    def _get_datasets(self, rgb=False):
        if self._tr_data is None:
            common = dict(margin=self.loss_fn.border, eval=False, rgb=False)
            common.update({k: v for k, v in self.data_conf.items() if k in ('noise_max', 'rnd_gain', 'image_size')})
            sconf = {k: v for k, v in self.data_conf.items() if k in ('max_sc', 'max_rot', 'max_shear', 'max_proj')}
            sconf.update({'max_tr': 0, 'max_rot': math.radians(sconf['max_rot'])})

            ds = []
            ds_counts = {}
            if 1:
                # n=8516
                ds_counts[BatvikSynthPairDataset] = 1
                ds.append(BatvikSynthPairDataset(self.data_conf['path'],
                                                 subset=('14', '18', '20', '24', '31', '33'), **common, **sconf))
            if 1:
                # give more weight to this dataset as it's much smaller (n=102)
                ds_counts[GoogleEarthPairDataset] = 8516 // 102
                ds.append(GoogleEarthPairDataset(self.data_conf['path'], **common))

            if 1:
                # n = 2270 (v4d)
                common['noise_max'] = 0.0
                common['rnd_gain'] = 1.0
                pconf = dict(fixed_ground_res=0.25,  # 0.25; None
                             resize_max_sc=1.0)
                ds_counts[BatvikPairDataset] = (2 * 8516) // 2270
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

            trn_set = RedundantlySampledDataset(trn_set,
                                                [ds_counts[fullset.get_dataset_at_index(trn_set.indices[i]).__class__]
                                                 for i in range(len(trn_set))])
            val_set = RedundantlySampledDataset(val_set,
                                                [ds_counts[fullset.get_dataset_at_index(val_set.indices[i]).__class__]
                                                 for i in range(len(val_set))])

            self._tr_data, self._val_data, self._test_data = map(self.wrap_ds, (trn_set, val_set, tst_set))
        return self._tr_data, self._val_data, self._test_data

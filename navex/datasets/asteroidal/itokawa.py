import math
import os

from .base import AsteroidImagePairDataset
from ..base import AugmentedPairDatasetMixin


class ItokawaPairDataset(AsteroidImagePairDataset, AugmentedPairDatasetMixin):
    def __init__(self, root='data', folder='itokawa', noise_max=0.20, rnd_gain=(0.5, 2), image_size=512,
                 margin=16, eval=False, rgb=False, npy=False):
        assert not npy, '.npy format not supported'
        assert not rgb, 'rgb images not supported'

        AugmentedPairDatasetMixin.__init__(self, noise_max=noise_max, rnd_gain=rnd_gain, image_size=image_size,
                                           margin=margin, max_sc=1.0, fill_value=0, eval=eval, rgb=False,
                                           blind_crop=True)
        AsteroidImagePairDataset.__init__(self, os.path.join(root, folder), transforms=self.transforms,
                                          trg_north_ra=math.radians(90.53), trg_north_dec=math.radians(-66.30),
                                          cam_axis=[0, 0, -1], cam_up=[-1, 0, 0])
        # axis ra & dec from https://science.sciencemag.org/content/312/5778/1347.abstract
        # frames:
        #   - https://sbnarchive.psi.edu/pds3/hayabusa/HAY_A_AMICA_3_AMICAGEOM_V1_0/catalog/hayhost.cat
        #   - https://naif.jpl.nasa.gov/pub/naif/pds/data/hay-a-spice-6-v1.0/haysp_1000/data/ik/amica31.ti
        #
        # TODO: is it possible to determine trg body orientation in sc frame based on *.xyz.exr?
        #   - itokawa sc ori is frequently erroneus?
        #   - eros sc ori missing, would need to use spice otherwise

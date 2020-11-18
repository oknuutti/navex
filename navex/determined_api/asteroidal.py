from determined.pytorch import PyTorchTrialContext

from ..trials.asteroidal import AsteroidalTrial
from .base import TrialWrapperBase


class AsteroidalTrialWrapper(TrialWrapperBase):
    def __init__(self, context: PyTorchTrialContext):
        trial = AsteroidalTrial(data_config=context.get_data_config(),
                                 batch_size=context.get_per_slot_batch_size(),
                                 **context.get_hparams())
        super(AsteroidalTrialWrapper, self).__init__(trial, context)

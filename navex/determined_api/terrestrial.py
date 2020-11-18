from determined.pytorch import PyTorchTrialContext

from ..trials.terrestrial import TerrestrialTrial
from .base import TrialWrapperBase


class TerrestrialTrialWrapper(TrialWrapperBase):
    def __init__(self, context: PyTorchTrialContext):
        optimizer_conf = context.get_hparam('optimizer')
        model_conf = context.get_hparam('model')
        loss_conf = context.get_hparam('loss')
        data_conf = context.get_data_config()

        trial = TerrestrialTrial(model_conf=model_conf, loss_conf=loss_conf, data_conf=data_conf,
                                 optimizer_conf=optimizer_conf, batch_size=context.get_per_slot_batch_size())

        super(TerrestrialTrialWrapper, self).__init__(trial, context)

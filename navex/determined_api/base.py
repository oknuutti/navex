
from typing import Tuple, cast

import torch
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, TorchData

from ..trials.terrestrial import TrialBase


class TrialWrapperBase(PyTorchTrial):
    def __init__(self, trial: TrialBase, context: PyTorchTrialContext):
        super(TrialWrapperBase, self).__init__(context)

        # Really need something like the following?
        # self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        # self.data_downloaded = False

        # Store trial context for later use.
        self.context = context

        # wrap this trial
        self.trial = trial
        self.trial.loss_backward_fn = self.context.backward
        self.trial.step_optimizer_fn = self.context.step_optimizer

        # Initialize the model and wrap it using self.context.wrap_model().
        self.model = self.context.wrap_model(self.trial.model)

        # Initialize the optimizer and wrap it using self.context.wrap_optimizer().
        self.optimizer = self.context.wrap_optimizer(self.trial.optimizer)

        self.lr_scheduler = None
        if self.trial.lr_scheduler is not None:
            self.lr_scheduler = self.context.wrap_lr_scheduler(self.trial.lr_scheduler)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int):
        data, labels = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        return {"loss": self.trial.train_batch(data, labels, epoch_idx, batch_idx)}

    def evaluate_batch(self, batch: TorchData):
        data, labels = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        validation_loss, accuracy = self.trial.evaluate_batch(data, labels)
        return {"validation_loss": validation_loss, "accuracy": accuracy}

    def build_training_data_loader(self):
        return self.trial.build_training_data_loader()

    def build_validation_data_loader(self):
        return self.trial.build_validation_data_loader()

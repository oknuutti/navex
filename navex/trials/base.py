import abc
from typing import Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

from ..models import tools


def _bare(val):
    if isinstance(val, Tensor):
        return val.item() if val.size() == 1 else val.detach().cpu().numpy()
    return val


class TrialBase(abc.ABC, torch.nn.Module):
    def __init__(self, model, loss_fn, optimizer_conf, lr_scheduler=None, aux_cost_coef=0.5, acc_grad_batches=1):
        super(TrialBase, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.aux_cost_coef = aux_cost_coef
        self.acc_grad_batches = acc_grad_batches
        self.loss_backward_fn = TrialBase._def_loss_backward_fn
        self.step_optimizer_fn = TrialBase._def_step_optimizer_fn
        self.optimizer = self.get_optimizer(**optimizer_conf) if isinstance(optimizer_conf, dict) else optimizer_conf
        self.hparams = {}


    @abc.abstractmethod
    def get_optimizer(self, **kwargs):
        raise NotImplemented()

    @staticmethod
    def _def_loss_backward_fn(loss: Tensor):
        loss.backward()

    @staticmethod
    def _def_step_optimizer_fn(optimizer: Optimizer):
        optimizer.step()
        optimizer.zero_grad()

    def train_batch(self, data: Tuple[Tensor, Tensor], labels: Tensor, epoch_idx: int, batch_idx: int):
        self.model.train()
        output1 = self.model(data[0])
        output2 = self.model(data[1])

        if self.model.aux_qty:
            loss = self.loss(output1[0], output2[0], labels)
            for i in range(1, len(output1)):
                loss = loss + self.aux_cost_coef * self.loss(output1[i], output2[i], labels)
            output1, output2 = output1[0], output2[0]
        else:
            loss = self.loss(output1, output2, labels)

        self.loss_backward_fn(loss)

        if (batch_idx+1) % self.acc_grad_batches == 0:
            self.step_optimizer_fn(self.optimizer)

        return loss, (output1, output2)

    def evaluate_batch(self, data: Tuple[Tensor, Tensor], labels: Tensor, **acc_conf):
        self.model.eval()
        with torch.no_grad():
            output1 = self.model(data[0])
            output2 = self.model(data[1])
            if self.model.aux_qty and self.model.training:
                output1, output2 = output1[0], output2[0]
            validation_loss = self.loss(output1, output2, labels)
            accuracy = self.accuracy(output1, output2, labels, **acc_conf)
        return validation_loss, accuracy, (output1, output2)

    def loss(self, output1: Tensor, output2: Tensor, labels: Tensor):
        assert self.loss_fn is not None, 'loss function not implemented'
        return self.loss_fn(output1, output2, labels)

    def accuracy(self, output1: Tensor, output2: Tensor, aflow: Tensor,
                 top_k=300, mutual=True, ratio=False, success_px_limit=12):

        des1, det1, qlt1 = output1
        des2, det2, qlt2 = output2
        _, _, H2, W2 = det2.shape

        yx1, conf1, descr1 = tools.detect_from_dense(des1, det1, qlt1, top_k=top_k)
        yx2, conf2, descr2 = tools.detect_from_dense(des2, det2, qlt2, top_k=top_k)

        # [B, K1], [B, K1], [B, K1], [B, K1, K2]
        matches, norm, mask, dist = tools.match(descr1, descr2, mutual=mutual, ratio=ratio)

        return tools.error_metrics(yx1, yx2, matches, mask, dist, aflow, (W2, H2), success_px_limit)

    @abc.abstractmethod
    def build_training_data_loader(self):
        raise NotImplemented()

    @abc.abstractmethod
    def build_validation_data_loader(self):
        raise NotImplemented()

    @abc.abstractmethod
    def build_test_data_loader(self):
        raise NotImplemented()



import random
import shutil
import os
import time
import csv
import sys

import numpy as np

import torch
from torch.backends import cudnn

from .experiments.parser import ExperimentConfigParser, to_dict
from .trials.terrestrial import TerrestrialTrial
from . import RND_SEED


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    args = config.training

    os.makedirs(args.output, exist_ok=True)

    # if don't call torch.cuda.current_device(), fails later with
    #   "RuntimeError: cuda runtime error (30) : unknown error at ..\aten\visnav\THC\THCGeneral.cpp:87"
    torch.cuda.current_device()
    use_cuda = torch.cuda.is_available() and 0
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # try to get consistent results across runs
    #   => currently still fails, however, makes runs a bit more consistent
    _set_random_seed()
    trial = TerrestrialTrial(to_dict(config.model), to_dict(config.loss),
                             to_dict(config.optimizer), to_dict(config.data),
                             args.batch_size, 1)

    # optionally resume from a checkpoint
    best_loss = float('inf')
    best_epoch = -1
    start_epoch = 0
    checkpoint = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            best_loss = checkpoint['best_loss']
            trial.model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            quit()

    # define overall training dataset, set output normalization, load model to gpu
    train_loader = trial.build_training_data_loader()
    val_loader = trial.build_validation_data_loader()
    test_loader = trial.build_test_data_loader()

    trial.to(device)
    # torch.autograd.set_detect_anomaly(True)

    # optimizer needs to be restored after model is set to correct device
    if checkpoint is not None:
        trial.optimizer.load_state_dict(checkpoint['optimizer'])

    # evaluate model only
    if args.evaluate:
        output = validate(test_loader, trial.model, device, args, return_output=True)
        with open('out.tsv', 'w') as fh:
            fh.write('\n'.join('\t'.join(str(c) for c in row) for row in output))
        return

    # training loop
    for epoch in range(start_epoch, args.epochs):

        # train for one epoch
        lss, acc = process(train_loader, trial, epoch, device, args)
        stats = np.zeros(16)
        stats[:6] = [epoch, lss.avg, *acc.avg]

        # evaluate on validation set
        if (epoch+1) % args.test_freq == 0:
            lss, acc = validate(val_loader, trial, device, args)
            stats[6:11] = [lss.avg, *acc.avg]

            # remember best loss and save checkpoint
            is_best = lss.avg < best_loss
            best_epoch = epoch if is_best else best_epoch
            best_loss = lss.avg if is_best else best_loss

            # save best model
            if is_best:
                _save_checkpoint({
                    'name': args.name,
                    'epoch': epoch + 1,
                    'loss': lss.avg,
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    'model_conf': trial.model.conf,
                    'model': trial.model.state_dict(),
                    'optimizer': trial.optimizer.state_dict(),
                }, True, args.output, args.name)
        else:
            is_best = False

        # maybe save a checkpoint even if not best model
        if (epoch+1) % args.save_freq == 0 and not is_best:
            _save_checkpoint({
                'name': args.name,
                'epoch': epoch + 1,
                'loss': lss.avg,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'model_conf': trial.model.conf,
                'model': trial.model.state_dict(),
                'optimizer': trial.optimizer.state_dict(),
            }, False, args.output, args.name)

        # evaluate on test set if best yet result on validation set
        if is_best:
            lss, acc = validate(test_loader, trial, device, args)
            stats[11:] = [lss.avg, *acc.avg]

        # add row to log file
        _save_log(stats, epoch == 0, args.output, args.name)

        # early stopping
        if args.early_stopping > 0 and epoch - best_epoch >= args.early_stopping:
            print('=====\nEARLY STOPPING CRITERION MET (%d epochs since best validation loss)' % args.early_stopping)
            break

        print('=====\n')

    if start_epoch == args.epochs or epoch + 1 == args.epochs:
        print('MAX EPOCHS (%d) REACHED' % args.epochs)
    print('BEST VALIDATION LOSS: %.3f' % best_loss)


def process(loader, trial, epoch, device, args, validate_only=False, return_output=False):
    data_time = Meter()
    batch_time = Meter()
    losses = Meter()
    accs = Meter(n=4, median=False)

    if return_output:
        outputs = []

    end = time.time()
    for i, ((img1, img2), target) in enumerate(loader):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        input = (img1, img2)
        target = target.to(device, non_blocking=True)

        # measure elapsed data loading time
        data_time.update(time.time() - end)
        end = time.time()

        if validate_only:
            loss, acc, output = trial.evaluate_batch(input, target, top_k=300, mutual=True,
                                                     ratio=False, success_px_limit=6)
        else:
            # train one batch
            loss, output = trial.train_batch(input, target, epoch, i)

            # measure accuracy and record loss
            with torch.no_grad():
                acc = trial.accuracy(*output, target, top_k=300, mutual=True, ratio=False, success_px_limit=6)

        accs.update(acc)
        losses.update(loss.detach().cpu().numpy())
        if return_output:
            for i in range(args.batch_size):
                outputs.append([[o[i:i+1, ...].detach().cpu().numpy() for o in o1] for o1 in output])

        # measure elapsed processing time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i+1 == len(loader):
            print((('Test [{1}/{2}]' if validate_only else 'Epoch: [{0}][{1}/{2}] ') +
                  ' Load: {data_time.avg:.3f} '
                  ' Proc: {batch_time.avg:.3f} '
                  ' Loss: {loss.avg:.4f} '
                  ' Tot: {acc[0]:.3f}% '
                  ' Inl: {acc[1]:.3f}% '
                  ' Dist: {acc[2]:.2f} '
                  ' mAP: {acc[3]:.3f}% '
                  ).format(epoch, i+1, len(loader), batch_time=batch_time,
                           data_time=data_time, loss=losses, acc=accs.avg * np.array([100, 100, 1, 100])))

        if 0 and i+1 >= 1:
            break

    return outputs if return_output else (losses, accs)


def validate(test_loader, trial, device, args, return_output=False):
    with torch.no_grad():
        result = process(test_loader, trial, None, device, args, validate_only=True, return_output=return_output)
    return result


def _set_random_seed(seed=RND_SEED): #, fanatic=False):
    # doesnt work even if fanatic & use_cuda
    # if fanatic:
    #     # if not disabled, still some variation between runs, however, makes training painfully slow
    #     cudnn.enabled = False       # ~double time
    # if use_cuda:
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True      # 7% slower
    cudnn.benchmark = False         # also uses extra mem if True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _filename_pid(filename, path, id):
    ext = filename.rfind('.')
    filename = (filename[:ext] + '_' + id + filename[ext:]) if len(id) > 0 else filename
    return os.path.join(path, filename)


def _scale_img(arr):
    a, b = np.min(arr), np.max(arr)
    return (255*(arr - a) / (b - a)).astype('uint8')


def _save_log(stats, write_header, path, name, filename='stats.csv'):
    with open(_filename_pid(filename, path, name), 'a', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # maybe write header
        if write_header:
            w.writerow([' '.join(sys.argv)])
            pf = ['_loss', '_tot', '_inl', '_dst', '_map']
            w.writerow(['epoch'] + ['tr' + a for a in pf] + ['val' + a for a in pf] + ['tst' + a for a in pf])

        # write stats one epoch at a time
        w.writerow(stats)


def _save_checkpoint(state, is_best, path, name, filename='checkpoint.pth.tar'):
    torch.save(state, _filename_pid(filename, path, name))
    if is_best:
        shutil.copyfile(_filename_pid(filename, path, name), _filename_pid('model_best.pth.tar', path, name))


class Meter(object):
    """ Stores current values and calculates stats """
    def __init__(self, median=False, n=1):
        self.default_median = median
        self.n = n
        self.reset()

    @property
    def pop_recent(self):
        if self.default_median:
            val = np.nanmedian(self.recent_values, axis=0)
        else:
            val = np.nanmean(self.recent_values, axis=0)
        self.recent_values = np.zeros((0, self.n))
        return val[0] if self.n == 1 else val

    @property
    def sum(self):
        t = np.nansum(self.values, axis=0)
        return t[0] if self.n == 1 else t

    @property
    def count(self):
        t = np.sum(np.logical_not(np.isnan(self.values)), axis=0)
        return t[0] if self.n == 1 else t

    @property
    def avg(self):
        t = np.nanmean(self.values, axis=0)
        return t[0] if self.n == 1 else t

    @property
    def median(self):
        t = np.nanmedian(self.values, axis=0)
        return t[0] if self.n == 1 else t

    def reset(self):
        self.recent_values = np.zeros((0, self.n))
        self.values = np.zeros((0, self.n))

    def update(self, val):
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        if isinstance(val, (list, tuple)):
            val = np.array(val)
        if not isinstance(val, np.ndarray):
            val = np.array([val])

        val = val.reshape((-1, self.n))
        self.recent_values = np.concatenate((self.recent_values, val), axis=0)
        self.values = np.concatenate((self.values, val), axis=0)


if __name__ == '__main__':
    main()

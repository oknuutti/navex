import os

import torch

from navex.lightning.base import TrialWrapperBase
from .experiments.parser import ExperimentConfigParser, to_dict
from .train_o import validate
from .models.r2d2orig import R2D2
from .trials.terrestrial import TerrestrialTrial


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    args = config.training

    use_cuda = torch.cuda.is_available() and args.gpu
    if not use_cuda:
        torch.set_num_threads(os.cpu_count()//2 - 1)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    if 0:
        #model = to_dict(config.model)
        model = R2D2(path=args.resume)
    else:
        light = TrialWrapperBase.load_from_checkpoint(args.resume, map_location="cuda:0" if use_cuda else "cpu")
        model = light.trial.model

    trial = TerrestrialTrial(model, to_dict(config.loss), None, to_dict(config.data), args.batch_size)
    trial.to(device)

    fst, rgb = trial.model, None
    while True:
        try:
            fst = next(fst.children())
        except:
            rgb = fst.in_channels == 3
            break

    test_loader = trial.build_test_data_loader(rgb=rgb)
    output = validate(test_loader, trial, device, args, return_output=False)
    print('%s' % (output,))


if __name__ == '__main__':
    main()
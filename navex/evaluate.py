import os

import torch

from .experiments.parser import ExperimentConfigParser, to_dict
from .train_o import validate
from .models.r2d2 import R2D2
from .trials.terrestrial import TerrestrialTrial


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    args = config.training

    #model = to_dict(config.model)
    model = R2D2()

    trial = TerrestrialTrial(model, to_dict(config.loss), None, to_dict(config.data), args.batch_size, None)

    use_cuda = torch.cuda.is_available() and 0
    if not use_cuda:
        torch.set_num_threads(os.cpu_count()//2 - 1)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    test_loader = trial.build_test_data_loader(rgb=True)
    trial.to(device)

    output = validate(test_loader, trial, device, args, return_output=False)
    print('%s' % (output,))


if __name__ == '__main__':
    main()
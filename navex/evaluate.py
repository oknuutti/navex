import os

import torch

from navex.models.tools import is_rgb_model, load_model
from .experiments.parser import ExperimentConfigParser, to_dict
from .train_o import validate
from .trials.terrestrial import TerrestrialTrial


def main():
    def_file = os.path.join(os.path.dirname(__file__), 'experiments', 'definition.yaml')
    config = ExperimentConfigParser(definition=def_file).parse_args()
    args = config.training

    use_cuda = torch.cuda.is_available() and args.gpu
    if not use_cuda:
        torch.set_num_threads(os.cpu_count()//2 - 1)

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = load_model(args.resume, device, model_only=True)
    rgb = is_rgb_model(model)

    trial = TerrestrialTrial(model, to_dict(config.loss), None, to_dict(config.data), args.batch_size)
    trial.to(device)
    test_loader = trial.build_test_data_loader(rgb=rgb)
    validate(test_loader, trial, device, args, return_output=False)


if __name__ == '__main__':
    main()

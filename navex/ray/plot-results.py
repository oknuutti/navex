import os
import argparse
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import skopt.plots as skplt

from navex.experiments.parser import split_double_samplers, ExperimentConfigParser, to_dict, flatten_dict
from navex.ray.base import MySkOptSearch


def main():
    parser = argparse.ArgumentParser('plot experiment results')
    parser.add_argument('--path', '-p', help="experiment directory")
    parser.add_argument('--config', '-c', help="config file if need to update searcher")
    args = parser.parse_args()

    search_alg = MySkOptSearch()
    if os.path.isdir(args.path):
        search_alg.restore_from_dir(args.path)
    else:
        search_alg.restore(args.path)

    X, y = search_alg._skopt_opt.Xi, search_alg._skopt_opt.yi
    len_sc = search_alg._skopt_opt.base_estimator_.kernel.k2.length_scale

    if search_alg._parameters is None or len(search_alg._parameters) == 0:
        search_alg._parameters = [
            'loss/det_n', 'loss/base',
            'optimizer/learning_rate', 'optimizer/weight_decay', 'optimizer/eps',
            'data/noise_max', 'data/rnd_gain']
        search_alg.save(os.path.join(args.path, 'new-searcher-state.pkl'))

    if 0 and args.config:
        def_file = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'definition.yaml')
        parser = ExperimentConfigParser(definition=def_file)
        parser.add_argument('--path')
        full_conf = to_dict(parser.parse_args())
        initial, hparams = split_double_samplers(full_conf['hparams'])
        hparams = flatten_dict(hparams, sep='/')
        hparams = OrderedDict([(p.replace('/', '.'), hparams[p]) for p in search_alg._parameters])
        search_alg = MySkOptSearch(space=hparams, metric=search_alg.metric, mode=search_alg.mode,
                                   points_to_evaluate=X, evaluated_rewards=y)

    res = search_alg._skopt_opt.get_result()

    if 1:
        skplt.plot_convergence(res)

    mpl.rcParams['font.size'] = 6
    skplt.plot_objective(res, dimensions=search_alg._parameters)
    plt.show()


if __name__ == '__main__':
    main()

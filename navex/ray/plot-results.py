import os
import argparse
from collections import OrderedDict

import numpy as np
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

    if args.config:
        # Massage old results so that can be used with new search config,
        # also show old results as if run with new config.
        # NOTE: the best X and y iterations might be missing as those experiments run the longest and are likely
        #       to not be ready yet

        def_file = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'definition.yaml')
        parser = ExperimentConfigParser(definition=def_file)
        parser.add_argument('--path')
        full_conf = to_dict(parser.parse_args())
        initial, hparams = split_double_samplers(full_conf['hparams'])
        hparams = flatten_dict(hparams, sep='.')
        hparams = OrderedDict(hparams)
        #hparams = OrderedDict([(p.replace('/', '.'), hparams[p]) for p in search_alg._parameters if p in hparams])

        # -- as search_alg.metric is not saved, can't do following check:
        # if search_alg.metric != full_conf['search']['metric'] or search_alg.mode != full_conf['search']['mode']:
        #     print('WARNING: new config has different search metric, new rewards NOT acquired from json files')

        defaults = {'data.max_rot': 0.0}  # TODO: remove hardcoding
        X = np.array(X)
        Xn = np.stack([X[:, search_alg._parameters.index(p.replace('.', '/'))]
                          if p.replace('.', '/') in search_alg._parameters
                          else np.ones((len(X),)) * defaults[p]
                       for p in hparams.keys()], axis=1)
        space = MySkOptSearch.convert_search_space(hparams)
        search_alg = MySkOptSearch(space=space, metric=full_conf['search']['metric'],
                                   mode=full_conf['search']['mode'], points_to_evaluate=Xn.tolist(),
                                   evaluated_rewards=y)
        search_alg.save(os.path.join(args.path, 'new-searcher-state.pkl'))

    res = search_alg._skopt_opt.get_result()

    if 1:
        skplt.plot_convergence(res)

    mpl.rcParams['font.size'] = 6
    skplt.plot_objective(res, dimensions=search_alg._parameters)
    plt.show()


if __name__ == '__main__':
    main()

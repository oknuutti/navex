import os
import math
import warnings
import argparse
from joblib import Parallel, delayed
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b
import skopt.plots as skplt
from skopt.space import Categorical, Integer

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
        hparams = flatten_dict(hparams, sep='/')
        hparams = OrderedDict(hparams)
        #hparams = OrderedDict([(p.replace('/', '.'), hparams[p]) for p in search_alg._parameters if p in hparams])

        # -- as search_alg.metric is not saved, can't do following check:
        # if search_alg.metric != full_conf['search']['metric'] or search_alg.mode != full_conf['search']['mode']:
        #     print('WARNING: new config has different search metric, new rewards NOT acquired from json files')

        if 0:
            y = search_alg.rescale_rewards(y)

        defaults = {'data/max_rot': 0.0}  # TODO: remove hardcoding
        X = np.array(X)
        Xn = [list(row) for i, row in enumerate(zip(*[parse_pval(X, search_alg, p)
                                                         if p in search_alg._parameters
                                                         else np.ones((len(X),)) * defaults[p]
                                                      for p in hparams.keys()])) if not np.isnan(y[i])]
        y = list(np.array(y)[~np.isnan(y)])

        space = MySkOptSearch.convert_search_space(hparams)
        search_alg = MySkOptSearch(space=space, metric=full_conf['search']['metric'],
                                   mode=full_conf['search']['mode'], points_to_evaluate=Xn,
                                   evaluated_rewards=y)
        search_alg.save(os.path.join(args.path, 'new-searcher-state.pkl'))

    fitted_model = search_alg._skopt_opt.models[-1]
    matern_gain = math.sqrt(fitted_model.kernel_.k1.k1.constant_value)
    matern_len_sc = fitted_model.kernel_.k1.k2.length_scale
    noise_sd = math.sqrt(fitted_model.noise_)

    hparam_gpspace_lensc = dict(zip(search_alg._parameters, matern_len_sc))
    print(f'Matern kernel weight: {matern_gain}**2, Gaussian noise: {noise_sd}**2')
    print(f'Length scales of normalized hyperparams: {hparam_gpspace_lensc}')

    res = search_alg._skopt_opt.get_result()
    best_idx = np.argmin(res.func_vals)
    best_hparams = dict(zip(search_alg._parameters, res.x))
    best_fval = res.fun
    print(f'Best trial is #{best_idx} with fval={best_fval}: {best_hparams}')

    best_mean_x, best_mean_fval = acq_max_mean(search_alg._skopt_opt.space, fitted_model)
    mean_maximizing_hparams = dict(zip(search_alg._parameters, best_mean_x))
    print(f'Max expected fval={best_mean_fval} with {mean_maximizing_hparams}')

    if 1:
        skplt.plot_convergence(res)

    mpl.rcParams['font.size'] = 6
    if 0:
        plot_dims = list(np.where(matern_len_sc < 99)[0])
    else:
        plot_dims = list(range(len(matern_len_sc)))
    axs = skplt.plot_objective(res, dimensions=np.array(search_alg._parameters)[plot_dims], plot_dims=plot_dims)

    if 0:
        # for some unknown reason works for r2d2 but for disk no
        add_marker(axs, np.array(best_mean_x)[plot_dims], color='b', linestyle=":", linewidth=1)

    plt.show()


def parse_pval(X, search_alg, pname):
    i = search_alg._parameters.index(pname)
    type = float
    if isinstance(search_alg._parameter_ranges[i], Categorical):
        type = search_alg._parameter_ranges[i].categories[0].__class__
    elif isinstance(search_alg._parameter_ranges[i], Integer):
        type = int
    return X[:, i].astype(type).tolist()


def acq_max_mean(space, model, n_points=10000, n_restarts=10, n_jobs=None):
    n_jobs = n_jobs or max(1, os.cpu_count() - 1)

    X0 = space.transform(space.rvs(n_samples=n_points, random_state=np.random.mtrand._rand))
    values = model.predict(X0)
    X0 = X0[np.argsort(values)[:n_restarts]]

    def costfun(x, model):
        mean, mean_grad = model.predict(np.atleast_2d(x), return_mean_grad=True)
        return mean, mean_grad

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=n_jobs)(
            delayed(fmin_l_bfgs_b)(
                costfun, x,
                args=(model,),
                bounds=space.transformed_bounds,
                approx_grad=False,
                maxiter=20)
            for x in X0)

    cand_xs = np.array([r[0] for r in results])
    cand_acqs = np.array([r[1] for r in results])
    gp_space_x = cand_xs[np.argmin(cand_acqs)]

    x = space.inverse_transform(gp_space_x.reshape((1, -1)))[0]
    y = np.min(cand_acqs)
    return x, y


def add_marker(axs, x, color='b', fill=False, size=100, marker="*", linestyle="--", linewidth=1.):
    scatter_kwargs = dict(edgecolors=color, facecolor=color if fill else 'None',
                          s=size, lw=0. if fill else linewidth, marker=marker)
    n_dims = len(axs)
    for i in range(n_dims):
        for j in range(n_dims):
            # lower triangle
            if i == j:
                axs[i, j].axvline(x[i], linestyle=linestyle, color=color, lw=linewidth)
            elif i > j:
                axs[i, j].scatter([x[j]], [x[i]], **scatter_kwargs)


if __name__ == '__main__':
    main()

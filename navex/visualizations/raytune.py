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
from navex.ray.base import SkOptSearchSH


def main():
    parser = argparse.ArgumentParser('plot experiment results')
    parser.add_argument('--path', '-p', help="experiment directory")
    parser.add_argument('--config', '-c', help="config file if need to update searcher")
    args = parser.parse_args()

    search_alg = SkOptSearchSH()
    if os.path.isdir(args.path):
        search_alg.restore_from_dir(args.path)
    else:
        search_alg.restore(args.path)

    X, y = search_alg._skopt_opt.Xi, search_alg._skopt_opt.yi
    si = getattr(search_alg._skopt_opt.base_estimator_, 'si', None)

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


        defaults = {'data/max_rot': 0.0}  # TODO: remove hardcoding
        X = np.array(X)
        Xn = [list(row) for i, row in enumerate(zip(*[parse_pval(X, search_alg, p)
                                                         if p in search_alg._parameters
                                                         else np.ones((len(X),)) * defaults[p]
                                                      for p in hparams.keys()])) if not np.isnan(y[i])]
        y = list(np.array(y)[~np.isnan(y)])

        if si is not None:
            steps = si
        else:
            steps = [{0: 300, 1: 900, 2: 2700, 3: 8100}[np.where(np.array([9, 3, -0.3, -2]) < yi)[0][0]] for yi in y]

        space = SkOptSearchSH.convert_search_space(hparams)
        search_alg = SkOptSearchSH(space=space, metric=full_conf['search']['metric'],
                                   mode=full_conf['search']['mode'], points_to_evaluate=Xn,
                                   evaluated_rewards=y, evaluated_steps=steps,
                                   length_scale_bounds=(0.1, 100))
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

    mpl.rcParams['font.size'] = 9
    mpl.rcParams['axes.labelsize'] = 11
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['lines.markersize'] = MARKER_SIZE
    # mpl.rcParams['lines.linewidth'] = LINE_WIDTH
    # plt.rc('axes', prop_cycle=default_cycler)
    # mpl.rcParams['legend.fontsize'] = 8

    if 1:
        plot_dims = list(np.where(matern_len_sc < 20)[0])
    else:
        plot_dims = list(range(len(matern_len_sc)))

    # hardcoded dim labels for each tune experiment
    cat_subs = {}
    if 'hafe_lr2d2' in args.path:
        label_map = {
            'loss/wdt': r'loss/$\alpha$',
            'loss/wpk': r'loss/$\beta$',             # ls=24
            'loss/det_n': 'loss/$n_{rep}$',         # ls=100
            'loss/base': r'loss/$\kappa$',
            'loss/sampler/pos_d': 'loss/$r_{pos}$',
            'loss/sampler/neg_d': 'loss/$r_{neg}$',
            'optimizer/weight_decay': 'opt/wd',
            'data/max_rot': r'data/$\lambda_r$',     # ls=46
            'data/max_proj': r'data/$\lambda_p$',
            'data/noise_max': r'data/$\lambda_n$',
            'data/use_synth': 'data/synth',
        }
    elif 'hafe_ldisk' in args.path:
        label_map = {
            'loss/wdt': r'loss/$\rho_{fp}$',             # ls=25
            'loss/det_n': 'loss/$h$',
            'loss/base': r'loss/$\theta_M$',
            'loss/sampler/pos_d': r'loss/$\epsilon$',
            'optimizer/weight_decay': 'opt/wd',
            'data/max_rot': r'data/$\lambda_r$',
            'data/max_proj': r'data/$\lambda_p$',   # ls=100
            'data/noise_max': r'data/$\lambda_n$',
            'data/use_synth': 'data/synth',
        }
    elif 'lafe' in args.path:
        arch_idx = search_alg._parameters.index('model/arch')
        cat_subs = {arch_idx: ['mn2', 'mn3', 'en0']}
        label_map = {
            'model/arch': 'model/arch',
            'model/des_head/use_se': 'model/des_se',
            'optimizer/weight_decay': 'opt/wd',
            'data/student_rnd_gain': r'data/$\lambda^{st}_g$',  # ls=100
            'data/student_noise_sd': r'data/$\lambda^{st}_\sigma$',  # ls=100
        }
    else:
        label_map = {}

    # transform dim labels and sort them
    dim_labels = [label_map.get(d, d) for d in np.array(search_alg._parameters)[plot_dims]]
    idxs = np.argsort(dim_labels)
    dim_labels = np.array(dim_labels)[idxs]
    plot_dims = np.array(plot_dims)[idxs]

    revmap1 = {i: j for j, i in enumerate(plot_dims)}
    revmap2 = {i: j for j, i in enumerate(idxs)}
    revmap = {i: revmap2[revmap1[i]] for i in revmap1}
    cat_subs = {revmap[i]: mapping for i, mapping in cat_subs.items() if i in revmap}

    # create result plot
    size = 1.6
    axs = skplt.plot_objective(res, size=size, dimensions=dim_labels, plot_dims=plot_dims.tolist())
    fig = plt.gcf()

    # add marker for surrogate model maximum mean
    add_marker(axs, best_mean_x, res.space, plot_dims, color='b', linestyle=":", linewidth=2.0)

    wide_ylab = False

    # adjust plot layout
    fw, fh = fig.get_size_inches()
    fig.set_size_inches(np.sqrt(2) * fh, fh)
    fw, fh = fig.get_size_inches()
    aw, ah = axs[0, 0].get_position().size
    lm, tm, rm, bm, hs, ws = (1.0 if wide_ylab else 0.8), 0.8, 0.42, 0.3, 0.1, 0.1  # in inches
    fig.subplots_adjust(left=lm/fw, right=1-rm/fw, bottom=bm/fh, top=1-tm/fh, wspace=ws/(fw*aw), hspace=hs/(fh*ah))

    # add extra margin into all plots
    add_plot_margins(axs, res.space, plot_dims, 0.05 * (ah / aw), 0.05)

    # adjust most labels
    tweak_labels(axs, 0.5 / (fw*aw), (0.7 if wide_ylab else 0.5) / (fw*aw), cat_subs)

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


def add_marker(axs, x, space, plot_dims, color='b', fill=False, size=100, marker="*", linestyle="--", linewidth=1.):
    x, space = zip(*[(x[i], space[int(i)][1]) for i in plot_dims])
    x = np.array(x)
    cI = [i for i, s in enumerate(space) if isinstance(s, Categorical)]
    x[cI] = np.array([space[i].categories.index(type(space[i].categories[0])(x[i])) for i in cI], dtype=float)

    scatter_kwargs = dict(edgecolors=color, facecolor=color if fill else 'None',
                          s=size, lw=0. if fill else 1.0, marker=marker)
    n_dims = len(axs)
    for i in range(n_dims):
        for j in range(n_dims):
            # lower triangle
            if i == j:
                axs[i, j].axvline(x[i], linestyle=linestyle, color=color, lw=linewidth)
            elif i > j:
                axs[i, j].scatter([x[j]], [x[i]], **scatter_kwargs)


def add_plot_margins(axs, space, plot_dims, xm, ym):
    space = [space[int(i)][1] for i in plot_dims]

    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            if isinstance(space[j], Categorical):
                xmin, xmax = 0, len(space[j].categories) - 1
            else:
                xmin, xmax = space[j].low, space[j].high

            if space[j].prior == 'log-uniform':
                xmarg = xm * (np.log(xmax) - np.log(xmin))
                ax.set_xlim(xmin / np.exp(xmarg), xmax * np.exp(xmarg))
            else:
                xmarg = xm * (xmax - xmin)
                ax.set_xlim(xmin - xmarg, xmax + xmarg)

            if i == j:
                ymin, ymax = ax.get_ylim()
            elif isinstance(space[i], Categorical):
                ymin, ymax = 0, len(space[i].categories) - 1
            else:
                ymin, ymax = space[i].low, space[i].high

            if i != j and space[i].prior == 'log-uniform':
                ymarg = ym * (np.log(ymax) - np.log(ymin))
                ax.set_ylim(ymin / np.exp(ymarg), ymax * np.exp(ymarg))
            else:
                ymarg = ym * (ymax - ymin)
                ax.set_ylim(ymin - ymarg, ymax + ymarg)


def tweak_labels(axs, near, far, cat_subs):
    for i, ax_row in enumerate(axs):
        if i > 0:
            ax = ax_row[0]

            if ax.yaxis.get_scale() != 'linear':
                ax.set_yscale('log')
                ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))

            if i % 2 == 1:
                # label closer to the plot
                ax.yaxis.set_label_coords(-near, 0.5)
            else:
                # label further away from the plot
                ax.yaxis.set_label_coords(-far, 0.5)


        for j, ax in enumerate(ax_row):
            if i in cat_subs and i != j:
                ax.set_yticks(np.arange(len(cat_subs[i])), cat_subs[i])
            if j in cat_subs:
                ax.set_xticks(np.arange(len(cat_subs[j])), cat_subs[j])
            if i == j:
                ax.set_ylabel("")
                if ax.xaxis.get_scale() != 'linear':
                    ax.set_xscale('log')
                    ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(numticks=999, subs="auto"))

    for j, ax in enumerate(axs[-1]):
        if j < len(axs[-1]) - 1:
            ax.set_xlabel("")
            ax.tick_params(axis='x', which='major', bottom=True, labelbottom=False)


if __name__ == '__main__':
    main()

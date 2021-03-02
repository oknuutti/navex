import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import skopt.plots as skplt

from navex.ray.base import MySkOptSearch


def main():
    parser = argparse.ArgumentParser('plot experiment results')
    parser.add_argument('--path', '-p', help="experiment directory")
    args = parser.parse_args()

    search_alg = MySkOptSearch()
    search_alg.restore_from_dir(args.path)
    X, y = search_alg._skopt_opt.Xi, search_alg._skopt_opt.yi
    len_sc = search_alg._skopt_opt.base_estimator_.kernel.k2.length_scale
    res = search_alg._skopt_opt.get_result()

    if 0:
        skplt.plot_convergence(res)

    if len(search_alg._parameters) == 0:
        search_alg._parameters = [
            'loss/det_n', 'loss/base',
            'optimizer/learning_rate', 'optimizer/weight_decay', 'optimizer/eps',
            'data/noise_max', 'data/rnd_gain']
        search_alg.save(os.path.join(args.path, 'new-searcher-state.pkl'))

    mpl.rcParams['font.size'] = 6
    skplt.plot_objective(res, dimensions=search_alg._parameters)
    plt.show()


if __name__ == '__main__':
    main()

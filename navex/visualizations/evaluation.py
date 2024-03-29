import argparse
import math
import os

from tqdm import tqdm
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

from ..datasets import tools
from ..evaluate import HEADER

# import warnings
# warnings.filterwarnings("error")

LEGACY = False


def main():
    parser = argparse.ArgumentParser("write here script description")
    parser.add_argument("--path", type=str, required=True, help='path to results csv file generated by evaluate.py')
    parser.add_argument("--dataset", "-d", choices=('eros', 'ito', '67p', 'synth'), action='append',
                        help='plot only results for given dataset')
    parser.add_argument("--joint-plot", type=int, default=1, help='Use all datasets for a joint plot, default: 1')
    args = parser.parse_args()

    selected_datasets = args.dataset or ['eros', '67p', 'ito', 'synth']

    rawdata = []
    with open(args.path) as fh:
        mapping = None
        for line in fh:
            fields = line.strip().split('\t')
            if mapping is None:
                if len(fields) > 20 and set(fields).issubset(HEADER):
                    mapping = dict([(h, fields.index(h)) for h in HEADER if h in fields])
            else:
                rawdata.append([fields[mapping[h]] if h in mapping else np.nan for h in HEADER])
    assert mapping is not None, 'could not find header row in file %s' % (args.path,)
    assert len(rawdata) > 0, 'could not find any data rows in file %s' % (args.path,)
    assert HEADER == ['Dataset', 'aflow', 'img1', 'img2', 'light1_x', 'light1_y', 'light1_z', 'light2_x', 'light2_y',
                      'light2_z', 'rel_qw', 'rel_qx', 'rel_qy', 'rel_qz', 'rel_angle', 'rel_dist',  'FD', 'M-Score',
                      'MMA', 'LE', 'mAP', 'dist-err', 'lat-err', 'ori-err', 'est_qw', 'est_qx', 'est_qy', 'est_qz',
                      'dist'], \
           'Header from evaluate.py does not correspond to current code'

    rawdata = np.array(rawdata, dtype=object)
    datasets = np.unique(rawdata[:, 0]).tolist()
    print("following datasets found: %s" % (datasets,))
    print("plotting results for dataset: %s" % (selected_datasets,))

    ds_info = [(sd, *{
        'eros': ('Eros', 1, 0),
        '67p': ('67P/C-G', 1, 0),
        'ito': ('Itokawa', 1, 0),
        'synth': ('Synthetic', 0, 1),
    }[sd]) for sd in selected_datasets]

    info = rawdata[:, 0:4]
    light1 = rawdata[:, 4:7].astype(float)
    light2 = rawdata[:, 7:10].astype(float)
    rel_q = [np.quaternion(*q) for q in rawdata[:, 10:14].astype(float)]
    rel_angle = rawdata[:, 14].astype(float)
    rel_dist = rawdata[:, 15].astype(float)
    feat_density = rawdata[:, 16].astype(float)
    mscore = rawdata[:, 17].astype(float)
    mma = rawdata[:, 18].astype(float)
    locerr = rawdata[:, 19].astype(float)
    mAP = rawdata[:, 20].astype(float)
    dist_err = np.empty((info.shape[0], 0)) if LEGACY else rawdata[:, 21].astype(float)
    lat_err = np.empty((info.shape[0], 0)) if LEGACY else rawdata[:, 22].astype(float)
    ori_err = rawdata[:, 21 if LEGACY else 23].astype(float)
    est_q = [np.quaternion(*q) for q in (rawdata[:, 22:26] if LEGACY else rawdata[:, 24:28]).astype(float)]
    dist = np.empty((info.shape[0], 0)) if LEGACY else rawdata[:, 28].astype(float)

    pa_change, ld_change = [None] * 2
    if not np.all(np.isnan(light1)):
        pa_change, ld_change = lighting_change(light1, light2)

    plot_viewangle_metrics(ds_info, info, rel_angle, mscore, mma, locerr, mAP, dist_err, lat_err, ori_err,
                           joint_plot=args.joint_plot)
    plot_lightchange_metrics(ds_info, info, pa_change, ld_change, mscore, mma, locerr, mAP, dist_err, lat_err, ori_err,
                             joint_plot=args.joint_plot)
    plt.show()


def lighting_change(light1, light2):
    """
    calc changes in phase angle and light direction, assumes cam axis +x, up +z
    """
    assert light1.shape[1] == 3 and light1.shape[0] > 0, 'invalid light1 array, should be of shape (-1, 3)'
    assert light2.shape[1] == 3 and light2.shape[0] > 0, 'invalid light2 array, should be of shape (-1, 3)'

    ld1 = np.copy(light1)
    ld1[:, 0] = 0
    ld2 = np.copy(light2)
    ld2[:, 0] = 0
    ld_change = tools.angle_between_rows(ld1, ld2)

    pa1 = np.arctan2(light1[:, 0], np.linalg.norm(ld1, axis=1))
    pa2 = np.arctan2(light2[:, 0], np.linalg.norm(ld2, axis=1))
    pa_change = np.abs(pa2 - pa1)

    return pa_change/np.pi*180, ld_change/np.pi*180


def plot_viewangle_metrics(ds_info, info, x_va, mscore, mma, locerr, mAP, disterr, laterr, orierr, joint_plot):
    xmin, xmax = 10, 30  #np.min(x_va), np.max(x_va)
    lims = np.arange(xmin, xmax + 5, 5)
    group_I = [np.logical_and(x_va >= lims[i], x_va < lims[i + 1]) for i in range(len(lims) - 1)]

    if np.sum([np.sum(group_I[i]) for i in range(len(group_I))]) == 0:
        if 0:
            # print("No values in range [%s, %s], removing limit" % (xmin, xmax))
            group_I = [np.ones(len(x_va), dtype=bool)]
            lims = np.array([0, 30])
        else:
            # print("No values in range [%s, %s], skipping plotting" % (xmin, xmax))
            return

    mask_I = np.ones(len(info), dtype=bool)
    for ds, ds_name, plt_view, plt_illu in ds_info:
        ds_I = info[:, 0] == ds
        mask_I[ds_I] = plt_view
        if plt_view:
            # print("\n===")
            # print("Stats vs view angle change:")
            print_stats(group_I[0][ds_I], mscore[ds_I], mma[ds_I], locerr[ds_I], mAP[ds_I], disterr[ds_I],
                        laterr[ds_I], orierr[ds_I], ds_name)

    if joint_plot:
        _plot_viewangle_metrics(group_I, mask_I, lims, mscore, mma, locerr, mAP, disterr, laterr, orierr)
    else:
        for ds, ds_name, plt_view, plt_illu in ds_info:
            ds_I = info[:, 0] == ds
            if plt_view:
                _plot_viewangle_metrics(group_I, ds_I, lims, mscore, mma, locerr, mAP,
                                        disterr, laterr, orierr, ds_name)


def _plot_viewangle_metrics(group_I, mask_I, lims, mscore, mma, locerr, mAP, disterr, laterr, orierr, ds_name=None):

    def plot(ax, y, max_val=None, points=30):
        nn = np.logical_not(np.logical_or(np.isnan(y), np.isinf(y)))
        ax.violinplot([y[np.logical_and.reduce((group_I[i], mask_I, nn))]
                       for i in range(len(group_I))], lims[:-1] + 2.5, widths=2.5,
                      showmedians=True, showextrema=False, points=points)
                      # quantiles=[[0.05], [0.5], [0.95]])
        if max_val is not None:
            ax.set_ylim(0, max_val)
        # ax.set_xlabel(xlabel)

    fig, axs = plt.subplots(2, 2, sharex=True, squeeze=True)
    fig.suptitle((f"{ds_name} m" if ds_name else "M") + "etrics vs angular camera axis change")  #, fontsize=16)
    axs = axs.flatten()

    axs[0].set_title('M-Score')
    plot(axs[0], mscore)

    axs[1].set_title('MMA')
    plot(axs[1], mma)

    axs[2].set_title('Pixel Loc. Error')
    plot(axs[2], locerr)

    if 0:
        axs[3].set_title('mAP')
        plot(axs[3], mAP)
    else:
        axs[3].set_title('Orientation Error')
        plot(axs[3], orierr, max_val=15, points=(30 * 180//15))

    plt.tight_layout()


def plot_lightchange_metrics(ds_info, info, x_pa, x_ld, mscore, mma, locerr, mAP, disterr, laterr, orierr, joint_plot=True):
    pa_min, pa_max = 0, 50
    ld_min, ld_max = 0, 80
    pa_lim = np.linspace(pa_min, pa_max, (pa_max - pa_min) // 10 + 1)
    ld_lim = np.linspace(ld_min, ld_max, (ld_max - ld_min) // 10 + 1)
    xx, yy = np.meshgrid(pa_lim, ld_lim)
    pa_step = np.kron(xx, np.ones((2, 2)))[1:-1, 1:-1]
    ld_step = np.kron(yy, np.ones((2, 2)))[1:-1, 1:-1]
    easy = np.logical_and.reduce((x_pa < 20, x_ld < 30))

    il_I = np.ones(len(info), dtype=bool)
    for ds, ds_name, plt_view, plt_illu in ds_info:
        ds_I = info[:, 0] == ds
        il_I[ds_I] = plt_illu
        if plt_illu:
            # print("\n===")
            # print("Stats vs lighting change:")
            print_stats(easy[ds_I], mscore[ds_I], mma[ds_I], locerr[ds_I], mAP[ds_I], disterr[ds_I],
                        laterr[ds_I], orierr[ds_I], ds_name)

    if joint_plot:
        _plot_lightchange_metrics(il_I, pa_step, ld_step, pa_lim, ld_lim, x_pa, x_ld, mscore, mma, locerr,
                                  mAP, disterr, laterr, orierr)
    else:
        for ds, ds_name, plt_view, plt_illu in ds_info:
            ds_I = info[:, 0] == ds
            if plt_illu:
                _plot_lightchange_metrics(ds_I, pa_step, ld_step, pa_lim, ld_lim, x_pa, x_ld, mscore, mma,
                                          locerr, mAP, disterr, laterr, orierr, ds_name)


def _plot_lightchange_metrics(mask_I, pa_step, ld_step, pa_lim, ld_lim, x_pa, x_ld, mscore, mma, locerr, mAP,
                              disterr, laterr, orierr, ds_name=None):
    fig = plt.figure()
    fig.suptitle((f"{ds_name} m" if ds_name else "M") + "etrics vs changes in lighting")  #, fontsize=16)

    fig2 = plt.figure()
    fig2.suptitle((f"{ds_name} s" if ds_name else "S") + "amples in grid")  #, fontsize=16)

    def plot(ax_i, z, zlabel=None, title=None):
        y_grouped = [[z[np.logical_and.reduce((
            mask_I,
            ~np.isnan(z),
            x_pa > pa_lim[i],
            x_pa < pa_lim[i + 1],
            x_ld > ld_lim[j],
            x_ld < ld_lim[j + 1],
        ))] for i in range(len(pa_lim) - 1)]
                for j in range(len(ld_lim) - 1)]

        medians = np.array([[np.median(y_grouped[j][i])
                                for i in range(len(pa_lim) - 1)]
                                    for j in range(len(ld_lim) - 1)])
        samples = np.array([[len(y_grouped[j][i])
                                for i in range(len(pa_lim) - 1)]
                                    for j in range(len(ld_lim) - 1)])

        print(f"samples: {np.sum(samples)}")

        for f, zz, zl in ((fig, medians, zlabel), (fig2, samples, 'samples')):
            zstep = np.kron(zz, np.ones((2, 2)))
            scalarMap = plt.cm.ScalarMappable(norm=Normalize(vmin=np.nanmin(zz), vmax=np.nanmax(zz)),
                                              cmap=plt.cm.PuOr_r)

            ax = f.add_subplot(2, 2, ax_i, projection='3d')
            ax.plot_surface(pa_step, ld_step, zstep, facecolors=scalarMap.to_rgba(zstep), antialiased=True)
            ax.view_init(50, 20)  # was 50, 35
            ax.set_xlabel(r"$|\alpha|$")  # phase angle
            ax.set_ylabel(r"$|\beta|$")   # lighting direction
            if title:
                ax.set_title(title, y=-0.25 if ax_i > 2 else None)
            if zl:
                ax.set_zlabel(zlabel)

    plot(1, mscore, title='M-Score')
    plot(2, mma, title='MMA')
    plot(3, locerr, title='Pixel Loc. Error')
    if 0:
        plot(4, mAP, title='mAP')
    else:
        plot(4, orierr, title='Orientation Error')
    plt.tight_layout()


def print_stats(easy_I, mscore, mma, locerr, mAP, disterr, laterr, orierr, ds_name):
    hard_I = np.logical_not(easy_I)

    stats = []
    for metric in (mscore, mma, mAP, locerr):
        nn = np.logical_not(np.isnan(metric))
        stat_n = (np.sum(np.logical_and(easy_I, nn)),
                  np.sum(np.logical_and(hard_I, nn)))
        stat = (np.mean(metric[np.logical_and(easy_I, nn)]),
                np.mean(metric[np.logical_and(hard_I, nn)]))
        stats.append((stat_n, stat))
    s_mscore_n, s_mscore = stats[0]
    s_mma_n, s_mma = stats[1]
    s_mAP_n, s_mAP = stats[2]
    s_locerr_n, s_locerr = stats[3]

    nn = np.logical_not(np.isnan(orierr))
    # fails = np.isinf(orierr)
    fails = orierr > 20
    orierr[fails] = 999
    lv = np.quantile(orierr[nn], 0.995)
    orierr[fails] = np.inf
    no = np.logical_and(orierr < lv, nn)

    # 0: Orientation estimate all, 1: avail gt, 2: has estimate (err lt %.3f), 3: err lt 10deg
    orierr_n = ((np.sum(easy_I), np.sum(np.logical_and(easy_I, nn)), np.sum(np.logical_and(easy_I, no)),
                    np.sum(np.logical_and.reduce((easy_I, no, orierr < 10)))),
                (np.sum(hard_I), np.sum(np.logical_and(hard_I, nn)), np.sum(np.logical_and(hard_I, no)),
                    np.sum(np.logical_and.reduce((hard_I, no, orierr < 10)))))

    orierr_rate = (100 * np.sum(np.logical_and(easy_I, fails)) / orierr_n[0][1],
                   100 * np.sum(np.logical_and(hard_I, fails)) / orierr_n[1][1])

    with np.errstate(invalid='ignore'):
        orierr_qtls = (np.quantile(orierr[np.logical_and(easy_I, nn)], (0.5, 0.85)),
                       np.quantile(orierr[np.logical_and(hard_I, nn)], (0.5, 0.85)))

    # print(f'dataset\tsubset\tsamples\tmscore [%]\tmma\tmap\tlocerr\terr_rate [%]\torierr_q50 [deg]\torierr_q84.1 [deg]')
    for i, subset in enumerate(('easy', 'hard')):
        print(f'{ds_name}\t{subset}\t{s_mscore_n[i]}\t{100*s_mscore[i]:.3f}\t'
              # f'{100*s_mma[i]:.3f}\t{100*s_mAP[i]:.3f}\t{s_locerr[i]:.3f}\t'
              f'{orierr_rate[i]:.3f}\t{orierr_qtls[i][0]:.3f}\t{orierr_qtls[i][1]:.3f}')


if __name__ == '__main__':
    main()

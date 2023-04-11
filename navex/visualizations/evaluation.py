import argparse
import math
import os

from tqdm import tqdm
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

from ..evaluate import HEADER
from ..datasets import tools


def main():
    parser = argparse.ArgumentParser("write here script description")
    parser.add_argument("--path", type=str, required=True, help='path to results csv file generated by evaluate.py')
    parser.add_argument("--dataset", choices=('eros', 'ito', '67p', 'synth'), action='append',
                        help='plot only results for given dataset')
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
                      'MMA', 'LE', 'mAP', 'ori-err', 'est_qw', 'est_qx', 'est_qy', 'est_qz'], \
           'Header from evaluate.py does not correspond to current code'

    rawdata = np.array(rawdata, dtype=object)
    datasets = np.unique(rawdata[:, 0]).tolist()
    print("following datasets found: %s" % (datasets,))
    print("plotting results for dataset: %s" % (selected_datasets,))
    I = np.logical_or.reduce([rawdata[:, 0] == sd for sd in selected_datasets])

    info = rawdata[I, 0:4]
    light1 = rawdata[I, 4:7].astype(float)
    light2 = rawdata[I, 7:10].astype(float)
    rel_q = [np.quaternion(*q) for q in rawdata[I, 10:14].astype(float)]
    rel_angle = rawdata[I, 14].astype(float)
    rel_dist = rawdata[I, 15].astype(float)
    feat_density = rawdata[I, 16].astype(float)
    mscore = rawdata[I, 17].astype(float)
    mma = rawdata[I, 18].astype(float)
    locerr = rawdata[I, 19].astype(float)
    mAP = rawdata[I, 20].astype(float)
    orierr = rawdata[I, 21].astype(float)
    est_q = [np.quaternion(*q) for q in rawdata[I, 22:26].astype(float)]

    pa_change, ld_change = [None] * 2
    if not np.all(np.isnan(light1)):
        pa_change, ld_change = lighting_change(light1, light2)

    plot_viewangle_metrics(rel_angle, mscore, mma, locerr, mAP, orierr)
    if pa_change is not None:
        plot_lightchange_metrics(pa_change, ld_change, mscore, mma, locerr, mAP, orierr)

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


def plot_viewangle_metrics(x_va, mscore, mma, locerr, mAP, orierr):
    xmin, xmax = 10, 30  #np.min(x_va), np.max(x_va)
    lims = np.arange(xmin, xmax + 5, 5)
    I = [np.logical_and(x_va >= lims[i], x_va < lims[i + 1]) for i in range(len(lims) - 1)]

    if np.sum([np.sum(I[i]) for i in range(len(I))]) == 0:
        if 0:
            print("No values in range [%s, %s], removing limit" % (xmin, xmax))
            I = [np.ones(len(x_va), dtype=bool)]
            lims = np.array([0, 30])
        else:
            print("No values in range [%s, %s], skipping plotting" % (xmin, xmax))
            return

    print("Stats vs view angle change:")
    print_stats(I[0], mscore, mma, locerr, mAP, orierr)

    def plot(ax, y, max_val=None, points=30):
        nn = np.logical_not(np.logical_or(np.isnan(y), np.isinf(y)))
        ax.violinplot([y[np.logical_and(I[i], nn)] for i in range(len(I))], lims[:-1] + 2.5, widths=2.5,
                      showmedians=True, showextrema=False, points=points)
                      # quantiles=[[0.05], [0.5], [0.95]])
        if max_val is not None:
            ax.set_ylim(0, max_val)
        # ax.set_xlabel(xlabel)

    fig, axs = plt.subplots(2, 2, sharex=True, squeeze=True)
    fig.suptitle("Metrics vs angular camera axis change")  #, fontsize=16)
    axs = axs.flatten()

    axs[0].set_title('M-score')
    plot(axs[0], mscore)

    axs[1].set_title('MMA')
    plot(axs[1], mma)

    axs[2].set_title('Localization Error')
    plot(axs[2], locerr)

    if 0:
        axs[3].set_title('mAP')
        plot(axs[3], mAP)
    else:
        axs[3].set_title('Orientation Error')
        plot(axs[3], orierr, max_val=15, points=(30 * 180//15))

    plt.tight_layout()


def print_stats(easy_I, mscore, mma, locerr, mAP, orierr):
    hard_I = np.logical_not(easy_I)
    nn = np.logical_not(np.isnan(mscore))
    print('M-score all (n=%d) median, mean: ' % (np.sum(nn),)
          + ', '.join(map(lambda x: '%.4f' % x, (
                                np.median(mscore[nn]),
                                np.mean(mscore[nn]),
                          ))))
    print('M-score easy (n=%d) median, mean: ' % (np.sum(np.logical_and(easy_I, nn)),)
          + ', '.join(map(lambda x: '%.4f' % x, (
                                np.median(mscore[np.logical_and(easy_I, nn)]),
                                np.mean(mscore[np.logical_and(easy_I, nn)]),
                          ))))
    print('M-score hard (n=%d) median, mean: ' % (np.sum(np.logical_and(hard_I, nn)),)
          + ', '.join(map(lambda x: '%.4f' % x, (
                                np.median(mscore[np.logical_and(hard_I, nn)]),
                                np.mean(mscore[np.logical_and(hard_I, nn)]),
                          ))))

    nn = np.logical_not(np.isnan(orierr))
    fails = np.isinf(orierr)
    orierr[fails] = 999
    lv = np.quantile(orierr[nn], 0.995)
    orierr[fails] = np.inf

    no = np.logical_and(orierr < lv, nn)
    print('Orientation estimate, all, avail gt, has estimate (err lt %.3f), err lt 10deg: %s => %s => %s => %s' % (
        lv, len(nn), np.sum(nn), np.sum(no), np.sum(np.logical_and(no, orierr < 10))))
    print('Orientation estimate, easy, avail gt, has estimate (err lt %.3f), err lt 10deg: %s => %s => %s => %s' % (
        lv, np.sum(easy_I), np.sum(np.logical_and(easy_I, nn)), np.sum(np.logical_and(easy_I, no)),
        np.sum(np.logical_and.reduce((easy_I, no, orierr < 10)))))
    print('Orientation estimate, hard, avail gt, has estimate (err lt %.3f), err lt 10deg: %s => %s => %s => %s' % (
        lv, np.sum(hard_I), np.sum(np.logical_and(hard_I, nn)), np.sum(np.logical_and(hard_I, no)),
        np.sum(np.logical_and.reduce((hard_I, no, orierr < 10)))))

    print('Orientation error all (n=%d) median, mean: ' % (np.sum(nn),)
          + ', '.join(map(lambda x: '%.4f' % x, (
                                np.median(orierr[nn]),
                                np.mean(orierr[no]),
                          ))))
    print('Orientation error easy (n=%d) median, mean: ' % (np.sum(np.logical_and(easy_I, nn)),)
          + ', '.join(map(lambda x: '%.4f' % x, (
                                np.median(orierr[np.logical_and(easy_I, nn)]),
                                np.mean(orierr[np.logical_and(easy_I, no)]),
                          ))))
    print('Orientation error hard (n=%d) median, mean: ' % (np.sum(np.logical_and(hard_I, nn)),)
          + ', '.join(map(lambda x: '%.4f' % x, (
                                np.median(orierr[np.logical_and(hard_I, nn)]),
                                np.mean(orierr[np.logical_and(hard_I, no)]),
                          ))))

    all_fails, all_tot = np.sum(fails), np.sum(nn)
    easy_fails, easy_tot = np.sum(np.logical_and(easy_I, fails)), np.sum(np.logical_and(easy_I, nn))
    hard_fails, hard_tot = np.sum(np.logical_and(hard_I, fails)), np.sum(np.logical_and(hard_I, nn))

    print('Orientation est. failure rate, all (n=%d): %.2f%%' % (all_tot, 100 * all_fails / all_tot))
    print('Orientation est. failure rate, easy (n=%d): %.2f%%' % (easy_tot, 100 * easy_fails / easy_tot))
    print('Orientation est. failure rate, hard (n=%d): %.2f%%' % (hard_tot, 100 * hard_fails / hard_tot))


def plot_lightchange_metrics(x_pa, x_ld, mscore, mma, locerr, mAP, orierr):
    pa_min, pa_max = 0, 40
    ld_min, ld_max = 0, 60
    pa_lim = np.linspace(pa_min, pa_max, 5)
    ld_lim = np.linspace(ld_min, ld_max, 7)
    xx, yy = np.meshgrid(pa_lim, ld_lim)
    pa_step = np.kron(xx, np.ones((2, 2)))[1:-1, 1:-1]
    ld_step = np.kron(yy, np.ones((2, 2)))[1:-1, 1:-1]

    print("\n===")
    print("Stats vs lighting change:")
    easy = np.logical_and.reduce((x_pa < pa_lim[2], x_ld < ld_lim[3]))
    print_stats(easy, mscore, mma, locerr, mAP, orierr)

    fig = plt.figure()
    fig.suptitle("Metrics vs changes in lighting")  #, fontsize=16)

    fig2 = plt.figure()
    fig2.suptitle("Samples in grid")  #, fontsize=16)

    def plot(ax_i, z, zlabel=None, title=None):
        y_grouped = [[z[np.logical_and.reduce((
            np.logical_not(np.isnan(z)),
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

        for f, zz, zl in ((fig, medians, zlabel), (fig2, samples, 'samples')):
            zstep = np.kron(zz, np.ones((2, 2)))
            scalarMap = plt.cm.ScalarMappable(norm=Normalize(vmin=np.nanmin(zz), vmax=np.nanmax(zz)),
                                              cmap=plt.cm.PuOr_r)

            ax = f.add_subplot(2, 2, ax_i, projection='3d')
            ax.plot_surface(pa_step, ld_step, zstep, facecolors=scalarMap.to_rgba(zstep), antialiased=True)
            #ax.view_init(30, -60)
            ax.view_init(50, 35)
            ax.set_xlabel("d-PA")
            ax.set_ylabel("d-LD")
            if title:
                ax.set_title(title)
            if zl:
                ax.set_zlabel(zlabel)

    plot(1, mscore, title='M-score')
    plot(2, mma, title='MMA')
    plot(3, locerr, title='Localization Error')
    if 0:
        plot(4, mAP, title='mAP')
    else:
        plot(4, orierr, title='Orientation Error')
    plt.tight_layout()


if __name__ == '__main__':
    main()

import argparse
import math
import os

from tqdm import tqdm
import numpy as np
import quaternion
import matplotlib.pyplot as plt

from ..evaluate import HEADER


# TODO:
#  - map/m-score/mma/loc-err of HAFE/LAFE/akaze/(root-)sift/superpoint/r2d2/disk on different datasets
#      - dimensions: phase angle, light direction changes,
#                   - rel ori angle


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

    plot_metrics_1d(rel_angle, mscore, mma, locerr, mAP, title="Metrics vs angular camera axis change")
    if pa_change is not None:
        plot_metrics_2d(pa_change, ld_change, mscore, mma, locerr, mAP, title="Metrics vs changes in lighting")

    plt.tight_layout()
    plt.show()


def lighting_change(light1, light2):
    # TODO: calc changes in phase angle and light direction

    pa_change, ld_change = [None] * 2

    return pa_change, ld_change


def plot_metrics_1d(dim, mscore, mma, locerr, mAP, xlabel=None, title=None):
    fig, axs = plt.subplots(2, 2, sharex=True, squeeze=True)
    fig.suptitle(title)  #, fontsize=16)
    axs = axs.flatten()

    xmin, xmax = 10, 30  #np.min(dim), np.max(dim)
    lims = np.arange(xmin, xmax + 5, 5)
    I = [np.logical_and(dim >= lims[i], dim < lims[i+1]) for i in range(len(lims) - 1)]

    def plot(ax, y):
        nn = np.logical_not(np.isnan(y))
        ax.violinplot([y[np.logical_and(I[i], nn)] for i in range(len(I))], lims[:-1] + 2.5, points=30, widths=2.5,
                      showmedians=True, showextrema=False)  # , quantiles=[[0.05], [0.5], [0.95]])
        ax.set_xlabel(xlabel)

    axs[0].set_title('M-score')
    plot(axs[0], mscore)
    nn = np.logical_not(np.isnan(mscore))
    print('M-score easy median, mean; all median, mean:\n'
          + '\n'.join(map(str, (np.median(mscore[np.logical_and(I[0], nn)]),
                                np.mean(mscore[np.logical_and(I[0], nn)]),
                                np.median(mscore[nn]),
                                np.mean(mscore[nn]))
                          )))

    axs[1].set_title('MMA')
    plot(axs[1], mma)

    axs[2].set_title('Localization Error')
    plot(axs[2], locerr)

    axs[3].set_title('mAP')
    plot(axs[3], mAP)


def plot_metrics_2d(dim1, dim2, mscore, mma, locerr, map, xlabel=None, title=None):
    fig, axs = plt.subplots(2, 2, sharex=True, squeeze=True)

    # TODO: make plots


if __name__ == '__main__':
    main()

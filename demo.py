import sys
import math
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import test_recalls
from gt import weighted_dist, interval_search


fontsize = 44
ticksize = 40
labelsize = 38
legendsize = 35
plt.style.use("seaborn-white")
W = 12.0
H = 9.5


def _plot_setting():
    plt.yticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.gcf().set_size_inches(W, H)
    plt.subplots_adjust(
        top=0.976,
        bottom=0.141,
        left=0.133,
        right=0.988,
        hspace=0.2,
        wspace=0.2
    )


def random_color(seed):
    np.random.seed(int(seed + 1))
    return list(np.random.choice(range(256), size=3) / 256.0)


marker = itertools.cycle(
    ("x", ".", ",", "o",
     "v", "s", "P", "<",
     "1", "2", "3", "4")
)


def run():
    nb, nq, d = 100000, 100, 16

    xb = np.random.uniform(size=(nb, d))
    xq = np.random.uniform(size=(nq, d))
    rg = np.random.uniform(size=(nq, d)) + 0.2

    gt = interval_search(xb, xq, rg)

    for p in [1, 2, 4, 8, 16]:
        dist = weighted_dist(xb, xq, rg, p=p)

        ss = np.argsort(np.array(dist))
        sys.stderr.flush()
        print("order of p: {}".format(p))
        rcs, pcs = test_recalls(ss, gt)
        rcs, pcs = np.reshape(rcs, -1), np.reshape(pcs, -1)
        plt.scatter(pcs, rcs,
                    label='p={}'.format(p),
                    color=random_color(p),
                    marker=next(marker),
                    linewidth=2.5)
        print()

    _plot_setting()
    plt.legend(
        loc="center left",
        fontsize=legendsize
    )

    plt.xlabel("Precision", fontsize=fontsize)
    plt.ylabel("Recall", fontsize=fontsize)
    plt.show()


run()

import numpy as np
from gt import weighted_dist, interval_search


def range_search(xb_, q_,  rg_):
    dist_ = np.abs(xb_ - q_)
    return np.nonzero(np.all(dist_ < rg_, axis=1))[0]


def weighted_search(xb_, q_, w_, p_):
    if p_ == -1:
        distances = np.sum(np.abs(xb_ - q_) > 1 / w_, axis=1)
    else:
        distances = np.sum(np.multiply(np.abs(q_ - xb_) ** p_, w_ ** p_), axis=1)
    return distances


def run_():
    nb, nq, d = 100000, 100, 16

    xb = np.random.uniform(size=(nb, d))
    xq = np.random.uniform(size=(nq, d))
    rg = np.random.uniform(size=(nq, d)) + 0.2

    for p in [1, 2, 3]:
        weighted_re_dist = weighted_dist(xb, xq, rg, p=p)
        weighted_gt_dist = [weighted_search(xb, q_, 1.0 / rg_, p_=p)
                            for q_, rg_ in zip(xq, rg)]
        for i, j in zip(weighted_re_dist, weighted_gt_dist):
            assert np.sum(np.abs(np.array(i) - np.array(j))) < 0.01

    res = interval_search(xb, xq, rg)
    gt = [range_search(xb, q_, rg_)
          for q_, rg_ in zip(xq, rg)]

    for i, j in zip(res, gt):
        assert np.all(np.array(i) == np.array(j))


run_()
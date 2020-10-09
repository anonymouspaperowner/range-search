import numpy as np
from scipy.cluster.vq import _vq
from vq_lp import vq_lp, lp_update_centroids


def run_():
    nb, nq, d = 100000, 100, 16
    ks = 256

    xs = np.random.uniform(size=(nb, d))
    centroids = np.random.uniform(size=(ks, d))

    codes_, dists_ = _vq.vq(xs, centroids)
    cb, _ = _vq.update_cluster_means(xs, codes_, ks)

    codes, dists = vq_lp(xs, centroids, p=2)
    lp_update_centroids(xs, centroids, codes, p=2)
    print("===================")
    print(cb)
    print(centroids)
    print("===================")
    print(codes_)
    print(codes)
    print("===================")
    print(dists_)
    print(dists)


run_()

import numpy as np
from . import cpp_vq_lp


def vq_lp(xs, centroids, p):
    assert xs.shape[1] == centroids.shape[1]
    codes = np.empty(shape=xs.shape[0], dtype=np.int32)
    dists = np.empty(shape=xs.shape[0], dtype=np.float)
    if xs.dtype == np.float32:
        cpp_vq_lp.vq_p_on_float(xs, centroids, codes, dists, p)
    elif xs.dtype == np.float64:
        cpp_vq_lp.vq_p_on_double(xs, centroids, codes, dists, p)
    else:
        assert False, "No implemented _vq_p " \
                      "for type {}".format(xs.dtype)
    return codes, np.power(dists, 1.0 / p)


def lp_update_centroids(xs, centroids, codes, p, iteration=10):
    assert xs.shape[1] == centroids.shape[1]
    assert xs.shape[0] == len(codes)
    if xs.dtype == np.float32:
        return cpp_vq_lp.update_centroids_on_float(
            xs, centroids, codes, p, iteration)
    elif xs.dtype == np.float64:
        return cpp_vq_lp.update_centroids_on_double(
            xs, centroids, codes, p, iteration)
    else:
        assert False, "No implemented _update_centroids " \
                      "for type {}".format(xs.dtype)

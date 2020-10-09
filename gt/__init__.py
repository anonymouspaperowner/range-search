import numpy as np
from . import cgt


def _interval_search_point(xb, xq, rg):
    assert len(xq) == len(rg)
    if xb.dtype == np.float32:
        return cgt.interval_search_on_float_point(xb, xq, rg)
    if xb.dtype == np.float64:
        return cgt.interval_search_on_double_point(xb, xq, rg)
    assert False, "No implemented interval_search_point " \
                  "for type {}".format(xb.dtype)


def _interval_search_interval(xb, xq, rg):
    assert len(xb) == len(rg)
    if xb.dtype == np.float32:
        return cgt.interval_search_on_float_interval(xb, xq, rg)
    if xb.dtype == np.float64:
        return cgt.interval_search_on_double_interval(xb, xq, rg)
    assert False, "No implemented interval_search_interval " \
                  "for type {}".format(xb.dtype)


def interval_search(xb, xq, rg):
    def _copy(res):
        return res
    if len(xq) == len(rg):
        return _copy(_interval_search_point(xb, xq, rg))
    if len(xb) == len(rg):
        return _copy(_interval_search_interval(xb, xq, rg))
    assert False, "shape of rg: {} mismatched with " \
                  "xb: {} and xq: {}".format(rg.shape, xb.shape, xq.shape)


def _interval_sparse_search_point(xb, xq, rg, idx):
    assert len(xq) == len(rg)
    if xb.dtype == np.float32:
        return cgt.sparse_interval_search_on_float_point(xb, xq, rg, idx)
    if xb.dtype == np.float64:
        return cgt.sparse_interval_search_on_double_point(xb, xq, rg, idx)
    assert False, "No implemented interval_search_point " \
                  "for type {}".format(xb.dtype)


def _interval_sparse_search_interval(xb, xq, rg, idx):
    assert len(xb) == len(rg)
    if xb.dtype == np.float32:
        return cgt.sparse_interval_search_on_float_interval(xb, xq, rg, idx)
    if xb.dtype == np.float64:
        return cgt.sparse_interval_search_on_double_interval(xb, xq, rg, idx)
    assert False, "No implemented interval_search_interval " \
                  "for type {}".format(xb.dtype)


def sparse_interval_search(xb, xq, rg, idx):
    def _copy(res):
        return res
    if len(xq) == len(rg):
        return _copy(_interval_search_point(xb, xq, rg, idx))
    if len(xb) == len(rg):
        return _copy(_interval_search_interval(xb, xq, rg, idx))
    assert False, "shape of rg: {} mismatched with " \
                  "xb: {} and xq: {}".format(rg.shape, xb.shape, xq.shape)


def _weighted_dist_point(xb, xq, rg, p):
    assert len(xq) == len(rg)
    if xb.dtype == np.float32:
        return cgt.weighted_dist_on_float_point(xb, xq, rg, p)
    if xb.dtype == np.float64:
        return cgt.weighted_dist_on_double_point(xb, xq, rg, p)
    assert False, "No implemented weighted_dist_point " \
                  "for type {}".format(xb.dtype)


def _weighted_dist_interval(xb, xq, rg, p):
    assert len(xb) == len(rg)
    if xb.dtype == np.float32:
        return cgt.weighted_dist_on_float_interval(xb, xq, rg, p)
    if xb.dtype == np.float64:
        return cgt.weighted_dist_on_double_interval(xb, xq, rg, p)
    assert False, "No implemented weighted_dist_interval " \
                  "for type {}".format(xb.dtype)


def weighted_dist(xb, xq, rg, p):
    def _copy(res):
        return np.reshape(res, (len(xq), len(xb)))
    if len(xb) == len(rg):
        return _copy(_weighted_dist_interval(xb, xq, rg, p))
    if len(xq) == len(rg):
        return _copy(_weighted_dist_point(xb, xq, rg, p))
    assert False, "shape of rg: {} mismatched with " \
                  "xb: {} and xq: {}".format(rg.shape, xb.shape, xq.shape)


def find_scale(xb, xq, rg, lower, upper):
    if xb.dtype == np.float32:
        return cgt.find_scale_on_float_interval(xb, xq, rg, lower, upper)
    if xb.dtype == np.float64:
        return cgt.find_scale_on_double_interval(xb, xq, rg, lower, upper)
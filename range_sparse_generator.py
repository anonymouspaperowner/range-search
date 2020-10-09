import os

import numpy as np

from vq import PQ
from lsh import SRP
from transform import transform
from scipy.spatial.distance import cdist

from utils import test_recalls
from gt import weighted_dist, sparse_interval_search, find_scale
from vecs_io import list2d_writer, list2d_reader
from vecs_io import fvecs_read, fvecs_writer, ivecs_writer


def _load_rg(seed, dataset, root, xb, xq, percent, rg_type, dim):
    nq_, d_ = xq.shape
    prefix = f"{root}/{dataset}/{dataset}"
    rg_file = f"{prefix}_s{seed}_p{percent}_{rg_type}_sparse_rg.fvecs"
    idx_file = f"{prefix}_s{seed}_p{percent}_{rg_type}_sparse_idx.txt"
    if not os.path.isfile(rg_file):
        np.random.seed(seed)
        rnd = np.random.uniform(low=0, high=1, size=(nq_, d_))
        idx = rnd.argsort(axis=1)[:, :dim].copy()
        idx.sort(axis=1)
        list2d_writer(idx_file, idx)

        np.random.seed(seed)
        lower = np.percentile(xb, q=50-percent/2., axis=0, keepdims=True)
        upper = np.percentile(xb, q=50+percent/2., axis=0, keepdims=True)
        if rg_type == 'uniform':
            rg_ = 1. + np.random.uniform(low=0., high=1., size=(nq_, d_))
        elif rg_type == 'normal':
            rg_ = np.random.normal(loc=0., scale=1., size=(nq_, d_))
            rg_ = 1. + np.abs(rg_)
        elif rg_type == 'zipf':
            rg_ = np.random.zipf(a=2., size=(nq_, d_))
        else:
            assert False, f"unknown range type {rg_type}"
        rg_ = rg_ * (upper - lower)
        fvecs_writer(rg_file, rg_)
    else:
        rg_ = fvecs_read(rg_file)
        idx = list2d_reader(idx_file)
    return rg_, idx


def _load_gt(seed, dataset, root, xb, xq, rg, idx, percent, rg_type):
    prefix = f"{root}/{dataset}/{dataset}"
    gt_file = f"{prefix}_s{seed}_p{percent}_{rg_type}_sparse_gt.txt"
    if not os.path.isfile(gt_file):
        print("# calculating gt")
        gt_ = sparse_interval_search(xb, xq, rg, idx)
        list2d_writer(gt_file, gt_)
    else:
        gt_ = list2d_reader(gt_file)
    sizes = [len(i) for i in gt_]
    print("# average size :", np.mean(sizes))
    print("# non empty average size :", np.mean([i for i in sizes if i > 0]))
    print("# non empty query :", np.count_nonzero(sizes))
    return gt_


def range_search(dataset, percent, rg_type, seed=808, root="../../data"):
    prefix = f"{root}/{dataset}/{dataset}"
    xb = fvecs_read(f"{prefix}_base.fvecs")
    xq = fvecs_read(f"{prefix}_query.fvecs")

    rg, idx = _load_rg(seed, dataset, root, xb, xq, percent, rg_type=rg_type)
    gt = _load_gt(seed, dataset, root, xb, xq, idx, percent, rg_type=rg_type)

    sizes = [len(i) for i in gt]
    return np.mean(sizes)


def find_percent(dataset, rg_type, return_size):
    print(f"find percent of {dataset} {rg_type} {return_size} ")
    lower = 0.
    upper = 100.
    for i in range(64):
        mid_percent = (lower + upper) / 2.
        avg_size = range_search(dataset=dataset, percent=mid_percent, rg_type=rg_type)
        diff = avg_size / return_size
        print(f"find percent of {dataset} {rg_type} {return_size} : {mid_percent} {avg_size} ")
        if 0.95 < diff < 1.05 or (upper - lower) < 10e-5:
            return mid_percent, avg_size
        if avg_size > return_size:
            upper = mid_percent
        elif avg_size < return_size:
            lower = mid_percent
    return mid_percent, avg_size


datasets = ["sift-128", "YearPredictionMSD", "glove-200", "nytimes-256", "gist-960", "deep-image-96"]
rg_types = ['uniform', 'normal', 'zipf']
rg_sizes = [1, 5, 10, 20, 50, 100, 1000]

with open("percent_record_sparse.csv", "w") as file_w:
    for dataset in datasets:
        for rg_type in rg_types:
            for return_size in rg_sizes:
                percent, true_size = find_percent(dataset, rg_type, return_size)
                print(dataset, rg_type, return_size, true_size, percent, sep=", ", end="\n========\n")
                print(dataset, rg_type, return_size, true_size, percent, sep=", ", end="\n", file=file_w)

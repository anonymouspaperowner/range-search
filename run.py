import os

import numpy as np

from vq import PQ
from lsh import SRP
from transform import transform
from scipy.spatial.distance import cdist

from utils import test_recalls
from gt import weighted_dist, interval_search, find_scale
from vecs_io import list2d_writer, list2d_reader
from vecs_io import fvecs_read, fvecs_writer, ivecs_writer


def _hamming_dist(q, x):
    return cdist(q, x, "hamming")


def srp_(tq_, tx_):
    nd = len(tx_[0])
    srp = SRP(k=256, d=nd)
    hx, hq = srp.hash(tx_), srp.hash(tq_)
    return _hamming_dist(hq, hx)


def l2_dist_(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]
    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * np.dot(q, x)
    l2[np.nonzero(l2 < 0)] = 0.0
    return np.sqrt(l2)


def vq_(tq_, tx_, M):
    qt = PQ(M=M, Ks=256, p=2)
    cx = qt.fit(tx_, iter=20).compress(tx_)
    return l2_dist_(tq_, cx)


def vq_nt_(q_, x_, rg_, p_, M, pq_lp):
    qt = PQ(M=M, Ks=256, p=pq_lp)
    cx = qt.fit(x_, iter=20).compress(x_)
    dist = weighted_dist(cx, q_, rg_, p=p_)
    return dist


def rg_by_percent(args):
    xb_, xq_, selectivity = args
    nq_ = len(xq_)
    rg_ = np.empty(shape=nq_)
    for qi in range(nq_):
        rg_[qi] = np.percentile(np.abs(xb_ - xq_[qi]), selectivity[qi])
    return rg_


def _load_rg(seed, dataset, root, xb, xq, percent, rg_type):
    nq_, d_ = xq.shape
    prefix = f"{root}/{dataset}/{dataset}"
    rg_file = f"{prefix}_s{seed}_p{percent}_{rg_type}_rg.fvecs"
    if not os.path.isfile(rg_file):
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
    return rg_


def _load_rg_depracted(seed, dataset, root, xb, xq, lower, upper, rg_type):
    nq_, d_ = xq.shape
    prefix = f"{root}/{dataset}/{dataset}"
    rg_file = f"{prefix}_s{seed}_l{lower}_u{upper}_{rg_type}_rg.fvecs"
    if not os.path.isfile(rg_file):
        np.random.seed(seed)
        if rg_type == 'uniform':
            rg_ = 1. + np.random.uniform(low=0., high=1., size=(nq_, d_))
        elif rg_type == 'normal':
            rg_ = np.random.normal(loc=0., scale=1., size=(nq_, d_))
            rg_ = 1. + np.abs(rg_)
        elif rg_type == 'zipf':
            rg_ = 1. + np.random.zipf(a=2., size=(nq_, d_))
        else:
            assert False, f"unknown range type {rg_type}"
        rg_ = rg_ * find_scale(xb, xq, rg_, lower, upper).reshape(nq_, 1)
        fvecs_writer(rg_file, rg_)
    else:
        rg_ = fvecs_read(rg_file)
    return rg_


def _load_gt(seed, dataset, root, xb, xq, rg, percent, rg_type):
    prefix = f"{root}/{dataset}/{dataset}"
    gt_file = f"{prefix}_s{seed}_p{percent}_{rg_type}_gt.txt"
    if not os.path.isfile(gt_file):
        print("# calculating gt")
        gt_ = interval_search(xb, xq, rg)
        list2d_writer(gt_file, gt_)
    else:
        gt_ = list2d_reader(gt_file)
    sizes = [len(i) for i in gt_]
    print("# average size :", np.mean(sizes))
    print("# non empty average size :", np.mean([i for i in sizes if i > 0]))
    print("# non empty query :", np.count_nonzero(sizes))
    return gt_


def run_inner_product(dataset, percent, rg_type, recall, seed=808, root="../../data"):
    # performance of random projection is bad
    # after transformed into inner product problem
    prefix = f"{root}/{dataset}/{dataset}"
    xb = fvecs_read(f"{prefix}_base.fvecs")
    xq = fvecs_read(f"{prefix}_query.fvecs")

    rg = _load_rg(seed, dataset, root, xb, xq, percent, rg_type=rg_type)
    gt = _load_gt(seed, dataset, root, xb, xq, rg, percent, rg_type=rg_type)
    if recall:
        scale = np.percentile(np.abs(xb), 75, axis=0, keepdims=True)
        xb /= scale
        xq /= scale
        rg /= scale

        xq = xq[:100]
        rg = rg[:100]
        gt = gt[:100]

        for p in [2, 4, 8, 16]:
            print("p = {}, ranked by p-dist".format(p))
            x, q = transform(xb, xq, rg, p=p, intervals=False)
            dist = -np.dot(q, x.T)
            test_recalls(np.argsort(dist), gt)

            print("p = {}, ranked by random projection".format(p))
            x, q = transform(xb, xq, rg, p=p, intervals=False)
            proj = np.random.normal(size=(x.shape[1], 32))
            x, q = np.dot(x, proj), np.dot(q, proj)
            dist = -np.dot(q, x.T)
            test_recalls(np.argsort(dist), gt)


def run_experiment(dataset, percent, rg_type, recall, seed=808, root="../../data"):
    prefix = f"{root}/{dataset}/{dataset}"
    xb = fvecs_read(f"{prefix}_base.fvecs")
    xq = fvecs_read(f"{prefix}_query.fvecs")

    rg = _load_rg(seed, dataset, root, xb, xq, percent, rg_type=rg_type)
    gt = _load_gt(seed, dataset, root, xb, xq, rg, percent, rg_type=rg_type)
    if recall:
        scale = np.percentile(np.abs(xb), 75, axis=0, keepdims=True)
        xb /= scale
        xq /= scale
        rg /= scale

        xq = xq[:100]
        rg = rg[:100]
        gt = gt[:100]

        # performance of random projection is bad
        # after transformed into inner product problem
        # for p in [2, 4, 8, 16]:
        #     print("p = {}, ranked by p-dist".format(p))
        #     x, q = transform(xb, xq, rg, p=p, intervals=False)
        #     dist = -np.dot(q, x.T)
        #     test_recalls(np.argsort(dist), gt)

        #     print("p = {}, ranked by random projection".format(p))
        #     x, q = transform(xb, xq, rg, p=p, intervals=False)
        #     proj = np.random.normal(size=(x.shape[1], 32))
        #     x, q = np.dot(x, proj), np.dot(q, proj)
        #     dist = -np.dot(q, x.T)
        #     test_recalls(np.argsort(dist), gt)

        for p in [2, 4, 8]:
            print("p = {}, ranked by p-dist".format(p))
            dist = weighted_dist(xb, xq, rg, p=p)
            test_recalls(np.argsort(dist), gt)

        for m in [2, 4, 8, 16]:
            for p in [2, 4, 8]:
                # print("pq_lp = p = {}, M = {} ranked by VQ-NT".format(p, m))
                # dist = vq_nt_(xq, xb, rg, p, M=m, pq_lp=p)
                # test_recalls(np.argsort(dist), gt)
                print("pq_lp = 2, p = {}, M = {} ranked by VQ-NT".format(p, m))
                dist = vq_nt_(xq, xb, rg, p, M=m, pq_lp=2)
                test_recalls(np.argsort(dist), gt)

    sizes = [len(i) for i in gt]
    return np.mean(sizes)


def save_pq(dataset, root="../data", seed_=808, percent=75, ms=[2, 32]):
    prefix = f"{root}/{dataset}/{dataset}"
    xb = fvecs_read(f"{prefix}_base.fvecs")
    scale = np.percentile(np.abs(xb), percent, axis=0, keepdims=True)
    xb /= scale
    for Ks in [256, 512]:
        for m in filter(lambda x: x < xb.shape[1], ms):
            codes_file = f"{prefix}_s{seed_}_pq{m}_ks{Ks}_codes.fvecs"
            centroids_file = f"{prefix}_s{seed_}_pq{m}_ks{Ks}_centroids.fvecs"
            scaling_file = f"{prefix}_s{seed_}_p{percent}_scale.fvecs"
            qt = PQ(M=m, Ks=Ks, p=2)
            qt.fit(xb, iter=20)
            if not os.path.isfile(centroids_file):
                codewords = qt.codewords.reshape(m * Ks, -1)
                print(f"codewords shape : {qt.codewords.shape}->{codewords.shape}")
                fvecs_writer(centroids_file, codewords)
            if not os.path.isfile(codes_file):
                codes = qt.encode(xb).astype(np.int32)
                print(f"codes shape : {codes.shape}")
                ivecs_writer(codes_file, codes)
            if not os.path.isfile(scaling_file):
                print(f"scale shape : {scale.shape}")
                fvecs_writer(scaling_file, scale)


datasets = ["sift-128", "YearPredictionMSD", "glove-200", "nytimes-256", "gist-960", "deep-image-96"]

#save_pq("gist-960", ms=[240])
save_pq("glove-200", ms=[50])
"""
for dataset in datsets:
    # save_pq(dataset=dataset)
    for percent in [45, 50, 55, 60, 65]:
        for rg_type in ['uniform', 'normal', 'zipf']:
            print(f"dataset={dataset}, percent={percent}, rg_type={rg_type}")
            run_experiment(dataset=dataset, percent=percent, rg_type=rg_type, recall=False)
        print()
"""


def find_percent(dataset, rg_type, return_size):
    print(f"find percent of {dataset} {rg_type} {return_size} ")
    lower = 0.
    upper = 100.
    for i in range(64):
        mid_percent = (lower + upper) / 2.
        avg_size = run_experiment(dataset=dataset, percent=mid_percent, rg_type=rg_type, recall=False)
        diff = avg_size / return_size
        print(f"find percent of {dataset} {rg_type} {return_size} : {mid_percent} {avg_size} ")
        if 0.95 < diff < 1.05 or (upper - lower) < 10e-5:
            return mid_percent, avg_size
        if avg_size > return_size:
            upper = mid_percent
        elif avg_size < return_size:
            lower = mid_percent
    return mid_percent, avg_size


with open("percent_record.csv", "w") as file_w:
    for dataset in datasets:
        for rg_type in ['uniform', 'normal', 'zipf']:
            for return_size in [1, 5, 10, 20, 50, 100, 1000]:
                percent, true_size = find_percent(dataset, rg_type, return_size)
                print(dataset, rg_type, return_size, true_size, percent, sep=", ", end="\n========\n")
                print(dataset, rg_type, return_size, true_size, percent, sep=", ", end="\n", file=file_w)


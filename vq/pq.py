from __future__ import division
from __future__ import print_function
import numpy as np
from .weighted_vq import weighted_kmeans
from vq_lp import vq_lp

def split_dim(d, m):
    reminder = d % m
    quotient = int(d / m)
    dims_width = [quotient + 1 if i < reminder
                  else quotient for i in range(m)]
    ds = np.cumsum(dims_width)  # prefix sum
    ds = np.insert(ds, 0, 0)  # insert zero at beginning
    return ds


class PQ(object):
    def __init__(self, M, Ks, p, verbose=True):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = (
            np.uint8
            if Ks <= 2 ** 8
            else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        )
        self.codewords = None
        self.Ds = None
        self.Dim = -1
        self.data_type = None
        self.p = p

    def class_message(self):
        return "Subspace PQ, M: {}, Ks : {}, code_dtype: {}".format(
            self.M, self.Ks, self.code_dtype
        )

    def fit(self, vecs, iter, w=None):
        assert vecs.ndim == 2
        N, D = vecs.shape
        self.data_type = vecs.dtype
        assert self.Ks < N, (
            "the number of training vector" " should be more than Ks"
        )
        self.Dim = D
        self.Ds = split_dim(self.Dim, self.M)
        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        codes_shape = (self.M, self.Ks, self.Ds[1])
        self.codewords = np.zeros(codes_shape, dtype=self.data_type)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m] : self.Ds[m + 1]]
            cw = self.codewords[m, :, : self.Ds[m + 1] - self.Ds[m]]
            cw[:, :] = weighted_kmeans(vecs_sub, w, self.Ks, self.p, iter, "points")[0]
        return self

    def encode(self, vecs):
        assert vecs.dtype == self.data_type
        assert vecs.ndim == 2
        N, D = vecs.shape

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m] : self.Ds[m + 1]]
            cw = self.codewords[m, :, : self.Ds[m + 1] - self.Ds[m]]
            codes[:, m], _ = vq_lp(vecs_sub, cw, p=self.p)

        return codes

    def decode(self, codes):
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Dim), dtype=self.data_type)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m] : self.Ds[m + 1]]
            vecs_sub[:, :] = self.codewords[
                m, codes[:, m], : self.Ds[m + 1] - self.Ds[m]
            ]

        return vecs

    def compress(self, vecs):
        return self.decode(self.encode(vecs))

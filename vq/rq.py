from __future__ import division
from __future__ import print_function
import numpy as np


class RQ(object):
    def __init__(self, pqs=None, verbose=True):

        assert len(pqs) > 0
        self.verbose = verbose
        self.depth = len(pqs)
        self.code_dtype = pqs[0].code_dtype
        self.M = max([pq.M for pq in pqs])
        self.pqs = pqs
        self.data_type = None

        for pq in self.pqs:
            assert pq.code_dtype == self.code_dtype

    def class_message(self):
        messages = ""
        for i, pq in enumerate(self.pqs):
            messages += pq.class_message()
        return messages

    def fit(self, x, iter, w=None):
        assert x.dtype == np.float32
        assert x.ndim == 2
        self.data_type = x.dtype

        vecs = np.empty_like(x)
        vecs[:, :] = x[:, :]

        for layer, pq in enumerate(self.pqs):
            pq.fit(vecs, iter, w=w)
            compressed = pq.compress(vecs)
            vecs = vecs - compressed
            del compressed
        return self

    def encode(self, vecs):
        """
        :param vecs:
        :return: (N * depth * M)
        """
        codes = np.zeros(
            (len(vecs), self.depth, self.M), dtype=self.code_dtype
        )  # N * deep * M
        for i, pq in enumerate(self.pqs):
            codes[:, i, : pq.M] = pq.encode(vecs)
            vecs = vecs - pq.decode(codes[:, i, : pq.M])
        return codes  # N * deep * M

    def decode(self, codes):
        vecs = [pq.decode(codes[:, i, : pq.M])
                for i, pq in enumerate(self.pqs)]
        return np.sum(vecs, axis=0)

    def compress(self, X):
        N, D = np.shape(X)

        sum_residual = np.zeros((N, D), dtype=X.dtype)

        vecs = np.zeros((N, D), dtype=X.dtype)
        vecs[:, :] = X[:, :]

        for i, pq in enumerate(self.pqs):
            compressed = pq.compress(vecs)
            vecs[:, :] = vecs - compressed
            sum_residual[:, :] = sum_residual + compressed
            del compressed

        return sum_residual

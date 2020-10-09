import warnings
import numpy as np
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans2


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    # divide by zero problem
    normalized_vecs = np.divide(vecs, norms_matrix,
                                out=np.zeros_like(vecs),
                                where=norms_matrix != 0)
    return norms, normalized_vecs


class NEQ(object):
    def __init__(self, ks, quantizer, true_norm=False,
                 verbose=True, method="kmeans", recover="quantize"):

        self.M = 2
        self.ks = ks
        self.true_norm = true_norm
        self.verbose = verbose
        self.method = method
        self.recover = recover
        self.code_dtype = (np.uint8 if ks <= 2 ** 8 else
                           (np.uint16 if ks <= 2 ** 16 else np.uint32))

        self.percentiles = None
        self.data_type = None
        self.quantizer = quantizer

    def class_message(self):
        return "NormPQ, percentiles: {}, quantize: {}".format(
            self.ks, self.quantizer.class_message()
        )

    def fit(self, vecs, iter, w=None):
        assert vecs.ndim == 2
        N, D = vecs.shape
        self.data_type = vecs.dtype
        assert self.ks < N, "the number of norm centroid " \
                            "should be more than Ks"

        norms, normalized_vecs = normalize(vecs)
        self.quantizer.fit(normalized_vecs, iter, w=w)

        if self.recover == "quantize":
            compressed_vecs = self.quantizer.compress(normalized_vecs)
            norms = norms / np.linalg.norm(compressed_vecs, axis=1)
        elif self.recover == "normalization":
            warnings.warn("Recover norm by normalization.")
            assert False
        else:
            warnings.warn("No normalization guarantee.")
            assert False

        if self.method == "kmeans":
            self.percentiles, _ = kmeans2(
                norms[:], self.ks, iter=iter, minit="points"
            )
        elif self.method == "kmeans_partial":
            indexes = np.argsort(norms)
            count = int(len(norms) * 0.7)
            centers_small_norms, _ = kmeans2(
                norms[indexes[:count]], self.ks // 2, iter=iter, minit="points"
            )
            centers_big_norms, _ = kmeans2(
                norms[indexes[count:]], self.ks // 2, iter=iter, minit="points"
            )
            self.percentiles = np.concatenate(
                (centers_small_norms, centers_big_norms)
            )

        elif self.method == "percentile":
            self.percentiles = np.percentile(
                norms, np.linspace(0, 100, self.ks + 1)[:]
            )
            self.percentiles = np.array(self.percentiles, dtype=self.data_type)
        elif self.method == "uniform":
            self.percentiles = np.linspace(
                np.min(norms), np.max(norms), self.ks + 1
            )
            self.percentiles = np.array(self.percentiles, dtype=self.data_type)
        elif self.method == "exponential":
            q = 0.98
            a = (1 - q) / (
                1 - q ** self.ks
            )  # make sure that sum of a*q**i is 1
            self.percentiles = [
                np.min(norms)
                if i == 0
                else np.min(norms)
                + a * (1 - q ** i) / (1 - q) * (np.max(norms) - np.min(norms))
                for i in range(self.ks + 1)
            ]

            self.percentiles = np.array(self.percentiles, dtype=self.data_type)
        else:
            assert False

        return self

    def encode_norm(self, norms):

        if self.method == "kmeans" or self.method == "kmeans_partial":
            norm_index, _ = vq(norms[:], self.percentiles)
        else:
            norm_index = [np.argmax(self.percentiles[1:] > n) for n in norms]
            norm_index = np.clip(norm_index, 1, self.ks)
        return norm_index

    def decode_norm(self, norm_index):
        if self.method == "kmeans" or self.method == "kmeans_partial":
            return self.percentiles[norm_index]
        else:
            return (
                self.percentiles[norm_index] + self.percentiles[norm_index - 1]
            ) / 2.0

    def compress(self, vecs):
        norms, normalized_vecs = normalize(vecs)

        compressed_vecs = self.quantizer.compress(normalized_vecs)
        del normalized_vecs

        if self.recover == "quantize":
            norms = norms / np.linalg.norm(compressed_vecs, axis=1)
        elif self.recover == "normalization":
            warnings.warn("Recover norm by normalization.")
            _, compressed_vecs = normalize(compressed_vecs)
            assert False
        else:
            warnings.warn("No normalization guarantee.")
            assert False

        if not self.true_norm:
            norms = self.decode_norm(self.encode_norm(norms))
        else:
            warnings.warn("Using true norm to compress vector.")
            assert False

        return (compressed_vecs.transpose() * norms).transpose()

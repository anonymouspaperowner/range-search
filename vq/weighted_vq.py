import warnings
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import vq
from scipy.cluster.vq import _vq
from scipy.cluster.vq import _valid_miss_meth
from scipy.cluster.vq import _valid_init_meth
from scipy.cluster.vq import _asarray_validated


def weighted_kmeans(data, w, k, p, iter=10,
                    minit='random', missing='warn',
                    check_finite=True, verbose=False):
    """
    Returns
    -------
    centroid : ndarray
        A 'k' by 'N' array of centroids found at the last iteration of
        k-means.
    label : ndarray
        label[i] is the code or index of the centroid the
        i'th observation is closest to.
    """
    n, d = data.shape
    if type(w) is np.ndarray:
        w = w.reshape((n, 1))

    assert int(iter) > 0, "Invalid iter (%s)" % iter
    miss_meth = _valid_miss_meth[missing]

    data = _asarray_validated(data, check_finite=check_finite)

    if data.size < 1:
        raise ValueError("Empty input is not supported.")

    # If k is not a single value it should be compatible with data's shape
    if minit == 'matrix' or not np.isscalar(k):
        code_book = np.array(k, copy=True)
        if data.ndim != code_book.ndim:
            raise ValueError("k array doesn't match data rank")
        nc = len(code_book)
        if data.ndim > 1 and code_book.shape[1] != d:
            raise ValueError("k array doesn't match data dimension")
    else:
        nc = int(k)

        if nc < 1:
            raise ValueError("Cannot ask kmeans2 for %d clusters"
                             " (k was %s)" % (nc, k))
        elif nc != k:
            warnings.warn("k was not an integer, was converted.")

        try:
            init_meth = _valid_init_meth[minit]
        except KeyError:
            raise ValueError("Unknown init method %r" % (minit,))
        else:
            code_book = init_meth(data, k)

    if p != 2:
        from vq_lp import vq_lp, lp_update_centroids
        for _ in tqdm(range(iter)) if verbose else range(iter):
            codes, dist = vq_lp(data, code_book, p=p)
            lp_update_centroids(data, code_book, codes, p=p)
        return code_book, vq_lp(data, code_book, p=p)[0]

    cluster_means = _vq.update_cluster_means
    for _ in tqdm(range(iter)) if verbose else range(iter):
        def _weighted_cluster_means(w_):
            new_code_book, has_members = cluster_means(data * w_, label, nc)
            density_sum, _ = cluster_means(w_, label, nc)
            new_code_book = np.divide(new_code_book,
                                      density_sum,
                                      where=density_sum != 0)
            return new_code_book, has_members
        # Compute the nearest neighbor for each obs using the current code book
        label, dist = vq(data, code_book)
        # Update the code book by computing centroids
        if w is None:
            new_code_book, has_members = cluster_means(data, label, nc)
        elif isinstance(w, str):
            if w == "cluster_size":
                # number of element in each cluster
                count = np.bincount(label, minlength=k)
                weight = np.reshape(count[label], (n, 1)).astype(data.dtype)
                new_code_book, has_members = _weighted_cluster_means(weight)
            if w == "cluster_radius":
                weight = np.reshape(dist, (n, 1))
                new_code_book, has_members = _weighted_cluster_means(weight)
        else:
            new_code_book, has_members = _weighted_cluster_means(w)
        if not has_members.all():
            miss_meth()
            # Set the empty clusters to their previous positions
            new_code_book[~has_members] = code_book[~has_members]
        code_book = new_code_book

    return code_book, _vq.vq(data, code_book)[0]

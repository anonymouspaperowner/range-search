import numpy as np
import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n-r)
    numerator = reduce(op.mul, range(n, n-r, -1), 1)
    denominator = reduce(op.mul, range(1, r+1), 1)
    return numerator / denominator


def scale_mean(x, q, w, scalar):
    mean = np.mean(x, axis=0, keepdims=True)
    x -= mean
    q -= mean
    scale = np.max(np.abs(x)) / scalar
    x /= scale
    q /= scale
    w /= scale
    return x, q, w


def simple_lsh(x, q):
    nx, d = np.shape(x)
    nq, _ = np.shape(q)
    x_ = np.empty(shape=(nx, d + 1))
    q_ = np.empty(shape=(nq, d + 1))

    x_[:, :d] = x
    q_[:, :d] = q

    norms = np.linalg.norm(x, axis=1)
    m = np.max(norms)
    x_[:, d] = np.sqrt(m**2 - norms**2)
    q_[:, d] = 0.0

    return x_, q_


def transform(x, q, w, p, intervals):
    assert p % 2 == 0

    xs = []
    qs = []

    for k in range(p+1):
        coefficient = - ncr(p, k) * (-1)**k
        xs.append(coefficient * x**(p - k))
        qs.append(q**k)

    w = w ** p
    if intervals:
        xs = [i / w for i in xs]
    else:
        qs = [i / w for i in qs]

    x = np.hstack(xs)
    q = np.hstack(qs)
    return x, q

import numpy as np
import struct


def ivecs_read(fname) -> np.ndarray:
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname) -> np.ndarray:
    return ivecs_read(fname).view('float32')


def bvecs_read(fname) -> np.ndarray:
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy()


# we mem-map the biggest files to avoid having them in memory all at
# once
def mmap_fvecs(fname) -> np.ndarray:
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]


def mmap_bvecs(fname) -> np.ndarray:
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def bvecs_read(filename) -> np.ndarray:
    return mmap_bvecs(fname=filename)


def fvecs_writer(filename, vecs):
    f = open(filename, "ab")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def ivecs_writer(filename, vecs):
    f = open(filename, "ab")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def list2d_writer(filename, vecs):
    with open(filename, 'a') as f:
        for vec in vecs:
            print(*vec, sep=' ', end='\n', file=f)


def list2d_reader(filename):
    with open(filename, 'r') as f:
        res = []
        for line in f.read().splitlines():
            if not line:
                res.append([])
            else:
                res.append(list(map(int, line.split(' '))))
        return res

import struct
import numpy as np
from vecs_io import fvecs_read


def srtree_writer(filename, vecs):
    f = open(filename, "w")
    dimension = len(vecs[0])
    print(dimension, end="\n", file=f)

    for i, x in enumerate(vecs):
        print(":".join(list(map(str, x))), end=":", file=f)
        print(f"({i})", end="\n", file=f)
    f.close()


for dataset in ["YearPredictionMSD", "sift-128"]:
    xb = fvecs_read(f"../../data/{dataset}/{dataset}_base.fvecs")
    xq = fvecs_read(f"../../data/{dataset}/{dataset}_query.fvecs")
    rg = fvecs_read(f"../../data/{dataset}/{dataset}_s808_p60_zipf_rg.fvecs")
    srtree_writer(f"../../data/{dataset}/{dataset}_base.rcd", xb)
    srtree_writer(f"../../data/{dataset}/{dataset}_query.rcd", xq)
    srtree_writer(f"../../data/{dataset}/{dataset}_s808_p60_zipf_rg.rcd", rg)


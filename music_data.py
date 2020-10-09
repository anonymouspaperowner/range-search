import numpy as np
from vecs_io import fvecs_writer

dataset = "YearPredictionMSD"

x = np.genfromtxt(f"/home/xydai/program/data/{dataset}/{dataset}.txt", delimiter=',')
np.random.shuffle(x)

nq = 10000
xq = x[:nq]
xb = x[nq:]
fvecs_writer(f"../../data/{dataset}/{dataset}_base.fvecs", xb)
fvecs_writer(f"../../data/{dataset}/{dataset}_query.fvecs", xq)

import numpy as np
import pandas as pd

from vecs_io import fvecs_writer


def load_uci(file, id_=None):
    x = pd.read_csv(file, delim_whitespace=True)
    if id_ is not None:
        x = x.set_index(id_)
    print(x.head(n=2))
    return x


root = "/home/xydai/program/data/"


def gas_sensor():
    dataset = "gas_sensor"
    data = load_uci(f"{root}/{dataset}/{dataset}_dataset.dat", id_="id")
    meta = load_uci(f"{root}/{dataset}/{dataset}_metadata.dat", id_="id")
    meta.drop('class', axis=1, inplace=True)
    meta.drop('date', axis=1, inplace=True)
    df = data.join(meta)
    print(df.head(n=2))
    x = df.to_numpy()
    print(x.shape)
    print(x[0, :])
    np.random.shuffle(x)

    nq = 10000
    fvecs_writer(f"{root}/{dataset}/{dataset}_base.fvecs", x[nq:, :])
    fvecs_writer(f"{root}/{dataset}/{dataset}_query.fvecs", x[:nq, :])


def covtype():
    dataset = "covtype"
    x = np.genfromtxt(f"{root}/{dataset}/{dataset}.data", delimiter=',')
    print(x.shape)
    np.random.shuffle(x)

    nq = 10000
    fvecs_writer(f"{root}/{dataset}/{dataset}_base.fvecs", x[nq:, :6])
    fvecs_writer(f"{root}/{dataset}/{dataset}_query.fvecs", x[:nq, :6])


covtype()

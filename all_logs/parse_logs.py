import numpy as np
import pandas as pd


def _parse_search_logs(files, columns_keys, columns_type, columns):
    assert len(columns_keys) == len(columns)
    assert len(columns_type) == len(columns)
    data = []
    columns_vals = [None] * len(columns)
    for filename in files:
        with open(filename, 'r') as f:
            if filename.startswith("tmp/build_rtree_d"):
                columns_vals[3] = int(filename[len("tmp/build_rtree_d"):len(filename)-len("/log.txt")])
            for line in f.read().splitlines():
                for i, key in enumerate(columns_keys):
                    if line.startswith(key):
                        columns_vals[i] = columns_type[i](line[len(key):])
                        if i + 1 == len(columns):
                            data.append(tuple(columns_vals))

    return pd.DataFrame.from_records(data, columns=columns)


def parse_brute_force(files):
    columns_keys = ['# dataset ',  '# rg size ', '# rg type ',
                    '# brute force search time ']

    columns_type = [str, int, str, float]
    columns = ['dataset', 'rg_size', 'rg_type', 'time']
    return _parse_search_logs(files, columns_keys, columns_type, columns)


def parse_tree(files):
    columns_keys = [
        '# dataset ', '# rg size ', '# rg type ', '# D ',
        'average insert time : ',
        'average probe time : ',
        'average prune time : ',
        'average total time : ',
        'average number of probed item : ',
    ]
    columns_type = [str, str, str, int, float, float, float, float, float]
    columns = [
        'dataset', 'rg_size', 'rg_type', 'D', 'insert_time',
        'probe_time', 'prune_time',  'total_time', 'probe_item'
    ]
    return _parse_search_logs(files, columns_keys, columns_type, columns)


def parse_lsh(files):
    columns_keys = [
        '# dataset ', '# L ', '# K ', '# R', '# rg size ', '# rg type ',
    ]
    columns_type = [str, str, str, str, str, str]
    columns = [
        'dataset', 'L', 'K', 'R', 'rg_size', 'rg_type',
        'probe_time', 'prune_time', 'total_time', 'probe_item',
        'average_recall', 'overall_recall',
        'average_precision', 'overall_precision'
    ]
    columns_vals = [None] * len(columns_type)

    data = []
    for filename in files:
        with open(filename, 'r') as f:
            for line in f.read().splitlines():
                for i, key in enumerate(columns_keys):
                    if line.startswith(key):
                        columns_vals[i] = columns_type[i](line[len(key):])
                if not line.startswith('#') and not line.startswith("avg probe time"):
                    numbers = list(map(float, line.split(" \t")))
                    data.append(tuple(columns_vals + numbers))

    return pd.DataFrame.from_records(data, columns=columns)


def parse_vq_exact(files):
    columns_keys = [
        '# dataset ', '# rg size ', '# rg type ',
    ]
    columns_type = [str, str, str]
    columns = [
        'dataset', 'rg_size', 'rg_type', 'total_time', 'recall', 'probe_item'
    ]
    columns_vals = [None] * len(columns_type)

    data = []
    for filename in files:
        with open(filename, 'r') as f:
            for line in f.read().splitlines():
                for i, key in enumerate(columns_keys):
                    if line.startswith(key):
                        columns_vals[i] = columns_type[i](line[len(key):])
                if line.startswith("exact 	time  : "):
                    numbers = [float(s) for s in line.split() if s.isdigit()]
                    data.append(tuple(columns_vals + numbers))

    return pd.DataFrame.from_records(data, columns=columns)

datasets = ['deepimage', 'gist', 'glove', 'nytime', 'sift', 'year_music']
# df = parse_vq_exact(
#     [f"tmp/vq/{i}/log_32_256.txt" for i in datasets]
# )
# df = df.sort_values(['dataset', 'rg_size', 'rg_type'])
# df.to_excel("log_pq.xlsx")

# datasets = ['deepimage', 'gist', 'glove', 'nytime', 'sift', 'year_music']
# df = parse_lsh(
#     [f"tmp/build_lsh_{i}/log.txt" for i in datasets]
# )
# df = df.sort_values(['dataset', 'rg_size', 'rg_type'])
# df.to_excel("log_lsh.xlsx")

# ds = [2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 64, 91, 128]
# files = [f"tmp/build_rtree_d{d}/log.txt" for d in ds]
# df = parse_tree(files)
# print(df)
# df.to_excel("log_rtree.xlsx")

# df = parse_brute_force(
#     [f"tmp/build_kdt_{i}/log.txt" for i in datasets]
# )
# df = df.sort_values(['dataset', 'rg_size', 'rg_type'])
# df.to_excel("log_bf.xlsx")
#
# df = parse_tree(
#     [f"tmp/build_kdt_{i}/log.txt" for i in datasets]
# )
# df.to_excel("log_kdt.xlsx")

pq = pd.read_excel("log_pq.xlsx")
bf = pd.read_excel("log_bf.xlsx")
for dataset in pq['dataset'].unique():
    for rg_size in pq['rg_size'].unique():
        for rg_type in pq['rg_type'].unique():
            query = f'(dataset=="{dataset}") & (rg_size == "{rg_size}") & (rg_type == "{rg_type}")'
            pq_i = pq.query(query)
            bf_i = bf.query(query)
            if len(pq_i) == 1 and len(bf_i) == 1:
                pq_t = pq_i.iloc[0]['total_time']
                bf_t = bf_i.iloc[0]['time']
                if bf_t < pq_t:
                    print(dataset, rg_type, rg_size, bf_t, pq_t)
            # else:
            #     print(dataset, rg_type, rg_size)



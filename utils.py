import sys
import math
import numpy as np


def test_recalls(sort, gt, file=sys.stdout):
    def intersect(gs, ids):
        recalls = []
        precisions = []
        for g, id in zip(gs, ids):
            if len(g) > 0:
                intersect_size = len(np.intersect1d(g, list(id)))
                recalls.append(intersect_size / len(g))
                precisions.append(intersect_size / len(id))
        return recalls, precisions

    ts = [2 ** i for i in range(2 + int(math.log2(len(sort[0]))))]
    rcs = []
    pcs = []
    print(" Probed \t Items \t recall\t precision", file=file)
    for t in ts:
        print("%6d \t %6d \t" % (t, len(sort[0, :t])), end="", file=file)
        rc, pc = intersect(gt, sort[:, :t])
        rcs.append(rc)
        pcs.append(pc)
        print("%.4f \t %.4f" % (np.mean(rc), np.mean(pc)), end="", file=file)
        print(file=file)
    return rcs, pcs

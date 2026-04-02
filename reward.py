import numpy as np


def mean_cosine(Xb, centroid):
    Xn = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12)
    cn = centroid / (np.linalg.norm(centroid) + 1e-12)
    raw = float(np.mean(np.dot(Xn, cn)))
    return (raw + 1) / 2


def running_mean(prev, Xb):
    return 0.9 * prev + 0.1 * np.mean(Xb, axis=0)
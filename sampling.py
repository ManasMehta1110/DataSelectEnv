import numpy as np


def entropy(proba):
    return -np.sum(proba * np.log(proba + 1e-12), axis=1)


def sim_to_noisy(X, centroid):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cn = centroid / (np.linalg.norm(centroid) + 1e-12)
    return np.dot(Xn, cn)


def sample_uncertainty(state, H_adj, n, exclude):
    if n <= 0:
        return []
    idx = np.argsort(-H_adj)
    return [i for i in idx if i not in exclude][:n]


def sample_diversity(state, n, exclude):
    if n <= 0:
        return []
    d = np.linalg.norm(state.X_pool - state.train_centroid, axis=1)
    idx = np.argsort(-d)
    return [i for i in idx if i not in exclude][:n]


def sample_random(state, n, exclude):
    if n <= 0:
        return []
    candidates = [i for i in range(len(state.X_pool)) if i not in exclude]
    if not candidates:
        return []
    return list(np.random.choice(candidates, size=min(n, len(candidates)), replace=False))
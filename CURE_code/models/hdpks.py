import numpy as np
from scipy.optimize import linear_sum_assignment
from .base import HDPModel

class HDPKS(HDPModel):
    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self, Xs, Ys, Xt, Yt=None):
        if self.normalize:
            Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
            Xt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-8)
        cost = np.linalg.norm(Xs.mean(0)[:, None] - Xt.mean(0)[None, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        self.mapping = col_ind
        return self

    def transform(self, X):
        return X[:, self.mapping]
import numpy as np
from .base import HDPModel

class CORAL(HDPModel):
    def __init__(self):
        pass

    def fit(self, Xs, Ys, Xt, Yt=None):
        self.Cs = np.cov(Xs.T) + np.eye(Xs.shape[1])
        self.Ct = np.cov(Xt.T) + np.eye(Xt.shape[1])
        return self

    def transform(self, X):
        Cs_inv_sqrt = np.linalg.inv(np.linalg.cholesky(self.Cs))
        Ct_sqrt = np.linalg.cholesky(self.Ct)
        return (X @ Cs_inv_sqrt) @ Ct_sqrt